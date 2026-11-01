package echo.collector

import android.content.Context
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import echo.shared.DepthDownsample
import java.io.File
import java.util.EnumSet

/**
 * ARCore backend — used only when the Camera2 path has no depth but
 * ARCore Depth is available (decided by the controller). ARCore owns
 * the camera in this mode (it cannot share it with a Camera2
 * session), so this backend produces everything: video (CPU images
 * encoded with MediaCodec), per-frame timestamps (frame.timestamp,
 * the same elapsedRealtime nanosecond domain as SensorEvent), 6DoF
 * poses (poses.csv sidecar, ARCore world frame — metres, +Y up,
 * right-handed) and depth downsampled to the 8x8 ToF grid.
 */
class ArCoreRecorder(
    private val context: Context,
    private val folder: File,
    private val onFrameStamp: (Long) -> Unit,
    private val onPose: (Long, Pose) -> Unit,
    private val onDepth: (Long, IntArray) -> Unit,
) {
    @Volatile private var running = false
    private var thread: Thread? = null
    private var session: Session? = null
    var note: String = "not started"
        private set
    var depthActive: Boolean = false
        private set

    fun start(onError: (String) -> Unit) {
        running = true
        thread = Thread {
            try {
                runLoop()
            } catch (e: Exception) {
                note = "arcore failed: ${e.message}"
                onError(note)
            }
        }.apply { start() }
    }

    private fun runLoop() {
        val sess = Session(context)
        session = sess
        val config = Config(sess)
        config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
        depthActive = sess.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
        if (depthActive) config.depthMode = Config.DepthMode.AUTOMATIC
        val filter = CameraConfigFilter(sess)
        filter.setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30))
        if (depthActive) {
            filter.setDepthSensorUsage(
                EnumSet.of(CameraConfig.DepthSensorUsage.REQUIRE_AND_USE))
        }
        val camConfigs = sess.getSupportedCameraConfigs(filter)
        val chosen = camConfigs.maxByOrNull {
            it.imageSize.width * it.imageSize.height
        } ?: throw IllegalStateException("no ARCore camera config")
        sess.cameraConfig = chosen
        sess.configure(config)
        sess.resume()

        val w = chosen.imageSize.width
        val h = chosen.imageSize.height
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, w, h)
        format.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
        )
        format.setInteger(MediaFormat.KEY_BIT_RATE, 10_000_000)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        val muxer = MediaMuxer(
            File(folder, "video.mp4").absolutePath,
            MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4
        )
        var track = -1
        var muxerStarted = false
        codec.start()
        val bufferInfo = MediaCodec.BufferInfo()
        var lastDepthNs = 0L
        var t0Ns: Long? = null

        fun drain(endOfStream: Boolean) {
            if (endOfStream) codec.signalEndOfInputStream()
            while (true) {
                val outIdx = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                when {
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        track = muxer.addTrack(codec.outputFormat)
                        muxer.start()
                        muxerStarted = true
                    }
                    outIdx >= 0 -> {
                        val buf = codec.getOutputBuffer(outIdx)!!
                        if (bufferInfo.size > 0 && muxerStarted) {
                            buf.position(bufferInfo.offset)
                            buf.limit(bufferInfo.offset + bufferInfo.size)
                            muxer.writeSampleData(track, buf, bufferInfo)
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            return
                        }
                    }
                    else -> {
                        if (endOfStream) continue else return
                    }
                }
                if (!endOfStream) return
            }
        }

        try {
            while (running) {
                val frame = try {
                    sess.update()
                } catch (e: CameraNotAvailableException) {
                    break
                }
                val ts = frame.timestamp
                if (ts == 0L) continue
                if (t0Ns == null) t0Ns = ts
                // --- video frame -----------------------------------
                val image = try {
                    frame.acquireCameraImage()
                } catch (e: NotYetAvailableException) {
                    continue
                }
                val inIdx = codec.dequeueInputBuffer(10_000)
                if (inIdx >= 0) {
                    val inBuf = codec.getInputBuffer(inIdx)!!
                    val nv12 = yuv420ToNv12(image)
                    inBuf.clear()
                    inBuf.put(nv12, 0, minOf(nv12.size, inBuf.remaining()))
                    codec.queueInputBuffer(
                        inIdx, 0, minOf(nv12.size, inBuf.capacity()),
                        (ts - t0Ns!!) / 1000, 0
                    )
                }
                image.close()
                onFrameStamp(ts)
                drain(false)
                // --- pose ------------------------------------------
                onPose(ts, frame.camera.pose)
                // --- depth ------------------------------------------
                if (depthActive && ts - lastDepthNs >= 100_000_000L) {
                    lastDepthNs = ts
                    try {
                        val dImg = frame.acquireDepthImage16Bits()
                        val plane = dImg.planes[0]
                        val shorts = ShortArray(plane.buffer.remaining() / 2)
                        plane.buffer.asShortBuffer().get(shorts)
                        if (plane.rowStride == dImg.width * 2) {
                            onDepth(ts, DepthDownsample.downsample(
                                shorts, dImg.width, dImg.height))
                        }
                        dImg.close()
                    } catch (e: NotYetAvailableException) {
                        /* no depth this frame yet */
                    }
                }
            }
        } finally {
            drain(true)
            try { codec.stop() } catch (e: Exception) { /* shutting down */ }
            codec.release()
            if (muxerStarted) muxer.stop()
            muxer.release()
            try { sess.pause() } catch (e: Exception) { /* shutting down */ }
            sess.close()
        }
        note = "arcore session ended cleanly (depth=$depthActive)"
    }

    fun stop() {
        running = false
        thread?.join(5_000)
        thread = null
    }

    /** YUV_420_888 -> NV12 byte array, honouring row/pixel strides. */
    private fun yuv420ToNv12(image: android.media.Image): ByteArray {
        val w = image.width
        val h = image.height
        val out = ByteArray(w * h * 3 / 2)
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        var pos = 0
        val yBuf = yPlane.buffer
        for (row in 0 until h) {
            for (col in 0 until w) {
                out[pos++] = yBuf.get(row * yPlane.rowStride + col * yPlane.pixelStride)
            }
        }
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        for (row in 0 until h / 2) {
            for (col in 0 until w / 2) {
                val uIdx = row * uPlane.rowStride + col * uPlane.pixelStride
                val vIdx = row * vPlane.rowStride + col * vPlane.pixelStride
                out[pos++] = uBuf.get(uIdx)
                out[pos++] = vBuf.get(vIdx)
            }
        }
        return out
    }
}
