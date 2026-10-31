package echo.collector

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.ImageReader
import android.media.MediaExtractor
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import android.util.Size
import android.view.Surface
import echo.shared.DepthDownsample
import java.io.File
import kotlin.math.roundToInt

/**
 * Camera2 backend: MediaRecorder H.264 -> video.mp4, per-frame
 * SENSOR_TIMESTAMPs harvested from capture callbacks (the frames'
 * own timestamps, never arrival times), and — when the chosen camera
 * exposes it — a DEPTH16 ImageReader downsampled to the 8x8 ToF grid.
 *
 * Focus: manual at the hyperfocal distance when the lens allows it
 * (focus breathing would move the intrinsics mid-capture — the same
 * reason the tethered build rejected autofocus UVC modules); falls
 * back to continuous-video AF otherwise, and the choice is recorded.
 */
class Camera2Recorder(
    private val context: Context,
    private val cameraId: String,
    private val folder: File,
    private val withDepth: Boolean,
    private val onFrameStamp: (Long) -> Unit,
    private val onDepth: (Long, IntArray) -> Unit,
) {
    private val cm = context.getSystemService(CameraManager::class.java)
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var recorder: MediaRecorder? = null
    private var depthReader: ImageReader? = null
    private var lastDepthNs = 0L
    var videoSize: Size? = null
        private set
    var focusNote: String = "unprobed"
        private set
    var depthSize: Size? = null
        private set
    var depthFovDeg: Double? = null
        private set

    @SuppressLint("MissingPermission") // camera permission is gated by the Activity
    fun start(onError: (String) -> Unit) {
        thread = HandlerThread("echo-cam2").apply { start() }
        handler = Handler(thread!!.looper)
        val ch = cm.getCameraCharacteristics(cameraId)
        val map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!

        // Video size: prefer 1920x1080, else the largest not exceeding it.
        val recSizes = map.getOutputSizes(MediaRecorder::class.java).toList()
        val size = recSizes.filter { it.width <= 1920 && it.height <= 1080 }
            .maxByOrNull { it.width * it.height }
            ?: recSizes.minByOrNull { it.width * it.height }!!
        videoSize = size

        val rec = MediaRecorder()
        recorder = rec
        rec.setVideoSource(MediaRecorder.VideoSource.SURFACE)
        rec.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        rec.setOutputFile(File(folder, "video.mp4").absolutePath)
        rec.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
        rec.setVideoSize(size.width, size.height)
        rec.setVideoFrameRate(30)
        rec.setVideoEncodingBitRate(12_000_000)
        rec.prepare()

        val surfaces = ArrayList<Surface>()
        surfaces.add(rec.surface)

        if (withDepth) {
            val dSizes = map.getOutputSizes(ImageFormat.DEPTH16).toList()
            val dSize = dSizes.maxByOrNull { it.width * it.height }
            if (dSize != null) {
                depthSize = dSize
                // This camera's own FOV (depth and colour are the same
                // physical camera on DEPTH16-capable devices): meta
                // records it as the ToF FOV so the importer's zone
                // geometry matches what produced the zones.
                val phys = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                val focals = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                if (phys != null && focals != null && focals.isNotEmpty()) {
                    depthFovDeg = Math.toDegrees(
                        2.0 * kotlin.math.atan((phys.width / 2.0) / focals.min())
                    )
                }
                val reader = ImageReader.newInstance(
                    dSize.width, dSize.height, ImageFormat.DEPTH16, 4
                )
                depthReader = reader
                reader.setOnImageAvailableListener({ r ->
                    val img = r.acquireLatestImage() ?: return@setOnImageAvailableListener
                    try {
                        val ts = img.timestamp
                        if (ts - lastDepthNs >= 100_000_000L) { // ~10 Hz cap
                            lastDepthNs = ts
                            val plane = img.planes[0]
                            val buf = plane.buffer
                            val shorts = ShortArray(buf.remaining() / 2)
                            buf.asShortBuffer().get(shorts)
                            // DEPTH16 images can carry row padding; the
                            // format's downsample assumes tight packing,
                            // which holds when rowStride == width*2.
                            if (plane.rowStride == img.width * 2) {
                                onDepth(ts, DepthDownsample.downsample(
                                    shorts, img.width, img.height))
                            }
                        }
                    } finally {
                        img.close()
                    }
                }, handler)
                surfaces.add(reader.surface)
            }
        }

        cm.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(dev: CameraDevice) {
                device = dev
                dev.createCaptureSession(
                    surfaces,
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(sess: CameraCaptureSession) {
                            session = sess
                            val req = dev.createCaptureRequest(
                                CameraDevice.TEMPLATE_RECORD
                            )
                            req.addTarget(rec.surface)
                            depthReader?.let { req.addTarget(it.surface) }
                            // FPS: lock to 30 where the device offers it.
                            val fpsRanges = ch.get(
                                CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES
                            ) ?: emptyArray()
                            fpsRanges.firstOrNull { it.lower == 30 && it.upper == 30 }
                                ?.let {
                                    req.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                                        Range(30, 30))
                                }
                            configureFocus(req, ch)
                            sess.setRepeatingRequest(
                                req.build(),
                                object : CameraCaptureSession.CaptureCallback() {
                                    override fun onCaptureCompleted(
                                        s: CameraCaptureSession,
                                        r: CaptureRequest,
                                        result: TotalCaptureResult,
                                    ) {
                                        result.get(CaptureResult.SENSOR_TIMESTAMP)
                                            ?.let { onFrameStamp(it) }
                                    }
                                },
                                handler
                            )
                            rec.start()
                        }

                        override fun onConfigureFailed(sess: CameraCaptureSession) =
                            onError("camera session configure failed")
                    },
                    handler
                )
            }

            override fun onDisconnected(dev: CameraDevice) { dev.close() }
            override fun onError(dev: CameraDevice, error: Int) {
                onError("camera error $error")
            }
        }, handler)
    }

    private fun configureFocus(req: CaptureRequest.Builder, ch: CameraCharacteristics) {
        val minFocus = ch.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
        if (minFocus > 0f) {
            // Manual focus exists. Hyperfocal H = f^2 / (N * c):
            // f = shortest focal (mm), N = fastest aperture, c = 2x
            // pixel pitch (mm) as the circle of confusion. Focus in
            // diopters (1/m) at H keeps H/2..infinity sharp.
            val f = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                ?.min() ?: 4f
            val n = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)
                ?.min() ?: 1.8f
            val phys = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            val px = ch.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            val pitchMm = if (phys != null && px != null && px.width > 0)
                phys.width / px.width else 0.0014f
            val hMm = (f * f) / (n * 2f * pitchMm)
            val diopters = (1000.0 / hMm).toFloat().coerceIn(0f, minFocus)
            req.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            req.set(CaptureRequest.LENS_FOCUS_DISTANCE, diopters)
            focusNote = "manual hyperfocal (%.2f diopters, H=%.1f m)".format(
                diopters, hMm / 1000.0)
        } else {
            req.set(
                CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
            )
            focusNote = "continuous-video AF (no manual focus on this lens)"
        }
    }

    /** Stops recording; returns (encodedFrameCount, note). The caller
     *  reconciles the harvested timestamps against the encoded count. */
    fun stop(): Pair<Int, String> {
        try { session?.stopRepeating() } catch (e: Exception) { /* tearing down */ }
        try { recorder?.stop() } catch (e: Exception) { /* no frames yet */ }
        recorder?.release(); recorder = null
        session?.close(); session = null
        depthReader?.close(); depthReader = null
        device?.close(); device = null
        thread?.quitSafely(); thread = null
        val video = File(folder, "video.mp4")
        if (!video.exists()) return 0 to "no video file"
        val ex = MediaExtractor()
        return try {
            ex.setDataSource(video.absolutePath)
            var count = 0
            for (i in 0 until ex.trackCount) {
                if (ex.getTrackFormat(i).getString("mime")?.startsWith("video/") == true) {
                    // Frame count via sample walk (cheap at these sizes).
                    ex.selectTrack(i)
                    while (ex.advance()) count++
                    break
                }
            }
            count to "encoded frames counted by extractor"
        } catch (e: Exception) {
            -1 to "extractor failed: ${e.message}"
        } finally {
            ex.release()
        }
    }
}
