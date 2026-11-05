package com.echo.collector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCrypto
import android.media.MediaFormat
import android.media.MediaMuxer
import android.media.MediaRecorder
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Surface
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.CameraIntrinsics
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.UnavailableException
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.util.Collections
import java.util.EnumSet

/**
 * One recording session. Owns the camera, the ARCore session, the sensors,
 * every CSV writer, and the stream watchdog. Lives inside
 * [RecorderService] (a foreground service) so it survives the screen
 * turning off and Android does not kill it mid-walk.
 *
 * ARCHITECTURE — reverse-engineered from the old EchoCollector (v0.14)
 * ----------------------------------------------------------------------
 * The old app's approach is the one that actually works on this phone:
 * ARCore OWNS the camera directly (a plain `new Session(context)`, NOT
 * the shared-camera API). ARCore renders camera frames to a GL texture;
 * the app pulls each frame with `frame.acquireCameraImage()`, converts
 * YUV to NV12, and encodes it with MediaCodec into video.mp4. Poses come
 * from the same loop, from `frame.camera.pose`.
 *
 * The shared-camera + Camera2 + MediaRecorder path (v2.0.0–2.0.3) is
 * abandoned: on this phone it silently produces no frames at all, so
 * every capture came out empty (0 B, 0 poses).
 *
 * THE OLD APP'S POSE BUG, AND HOW THIS CLASS FIXES IT
 * -----------------------------------------------------
 * The old app's poses died after ~6 s. Reverse-engineering found two
 * causes, and this class removes both:
 *
 * 1. A watchdog swapped ARCore -> Camera2 at 6 s whenever ARCore's frame
 *    rate was below 8 fps. Camera2 has no poses, so the route ended.
 *    FIX: there is NO mid-capture backend swap here. Once ARCore is
 *    running and producing poses, it stays. A Camera2 fallback exists
 *    only as a PRE-START decision (ARCore fails to bring up, or produces
 *    zero frames in the first 4 s) — never after poses exist.
 *
 * 2. Poses were written only `if (queued)` — i.e. only when the video
 *    encoder accepted the frame. The old encoder was slow (per-pixel
 *    YUV conversion, ~3M buffer reads per 1080p frame), so most frames
 *    were dropped and poses were dropped with them.
 *    FIX: poses are written for EVERY ARCore frame, unconditionally,
 *    before the encoder is touched. A slow encoder can drop video
 *    frames; it can never starve the pose stream. And the YUV->NV12
 *    conversion is bulk row copies (no per-pixel reads), at a capped
 *    1280x720 so the encoder keeps up.
 *
 * LENSES
 * ------
 * Main lens: ARCore (video + poses + sensors). Ultrawide lens: Camera2
 * only (video + sensors). ARCore can only track on the camera it chooses
 * (the main); the ultrawide cannot be used for tracking. This is
 * reported honestly in the UI and in meta.json (arcore_poses=false)
 * instead of silently producing an empty poses.csv.
 *
 * CLOCK
 * -----
 * Every timestamp in every file shares one clock:
 * SystemClock.elapsedRealtimeNanos(). t_s = (now - t0) / 1e9, where t0 is
 * taken once in start(). SensorEvent.timestamp, ARCore Frame.timestamp
 * and Camera2 SENSOR_TIMESTAMP are all in this same monotonic domain.
 */
class RecorderSession(
    private val context: Context,
    private val captureDir: File,
    private val lens: LensHelper.Lens,
    /** Called ~2x/second with the live status (for the UI + notification). */
    private val onStatus: (Status) -> Unit,
    /** Called when the watchdog finds a stalled stream. */
    private val onWarning: (String) -> Unit,
    /** Called with a small preview thumbnail (~5 fps) while recording. */
    private val onPreview: (Bitmap) -> Unit,
) {

    companion object {
        private const val TAG = "RecorderSession"
        /** Video is capped at 720p so the software YUV conversion + encoder
         *  keep up with 30 fps on this phone. The old app took the largest
         *  config (1080p) and fell below 8 fps, which tripped its watchdog. */
        private const val MAX_VIDEO_WIDTH = 1280
        private const val MAX_VIDEO_HEIGHT = 720
        private const val VIDEO_FPS = 30
        private const val VIDEO_BITRATE = 10_000_000
        private const val STATUS_INTERVAL_MS = 500L
        private const val FLUSH_INTERVAL_MS = 1000L
        /** If ARCore produces zero frames/poses in this long after start,
         *  it is dead; fall back to Camera2 before any pose exists. */
        private const val BRINGUP_TIMEOUT_MS = 4000L
        private const val PREVIEW_EVERY_N_FRAMES = 6
        private const val PREVIEW_MAX_DIM = 256
    }

    data class Status(
        val elapsedS: Double,
        val frameCount: Long,
        val poseCount: Long,
        val tracking: String,
        val lens: LensHelper.Lens,
        val arcoreActive: Boolean,
        val staleStreams: List<String>,
        /** "ARCore" or "普通相机" — so the screen always says which
         *  backend is actually running (the v2.1.0 screen never said). */
        val backend: String,
    )

    // ---- threads -----------------------------------------------------------
    // The status ticker runs on its OWN thread, so the UI always shows live
    // numbers even if the recorder thread is busy bringing up ARCore. (In
    // v2.1.0 the ticker shared the recorder thread, and the screen looked
    // frozen while ARCore started.)
    private val recorderThread = HandlerThread("recorder").apply { start() }
    private val recorderHandler = Handler(recorderThread.looper)
    private val statusThread = HandlerThread("status").apply { start() }
    private val statusHandler = Handler(statusThread.looper)
    private val sensorThread = HandlerThread("sensors").apply { start() }
    private val sensorHandler = Handler(sensorThread.looper)
    private val writerThread = HandlerThread("writer").apply { start() }
    private val writerHandler = Handler(writerThread.looper)

    // ---- session state -----------------------------------------------------
    @Volatile private var running = false
    @Volatile private var arcoreRunning = false
    private var t0Nanos: Long = -1
    private var startWallclockMs: Long = -1
    private var cameraId: String = ""
    private var lensInfo: LensHelper.LensInfo? = null
    private var videoSize = Size(MAX_VIDEO_WIDTH, MAX_VIDEO_HEIGHT)
    private var arcoreActive = false
    @Volatile private var trackingState: String = "UNKNOWN"
    @Volatile private var poseCount: Long = 0
    @Volatile private var frameCount: Long = 0
    @Volatile private var lastFrameTs: Double = 0.0
    private val staleStreams = Collections.synchronizedSet(mutableSetOf<String>())
    /** Human-readable backend note, recorded in meta.json (e.g. which
     *  backend ran and why, so a capture is diagnosable after the fact). */
    private var backendNote: String = "not started"
    /** Camera intrinsics from ARCore (fx, fy, cx, cy), captured once. */
    private var intrinsicsJson: JSONObject? = null

    // ---- ARCore backend (owns the camera) ----------------------------------
    private var arSession: Session? = null
    private var arcoreThread: Thread? = null
    private var eglDisplay: EGLDisplay? = null
    private var eglContext: EGLContext? = null
    private var eglSurface: EGLSurface? = null
    private var cameraTexId: Int = 0
    private var mediaCodec: MediaCodec? = null
    private var mediaMuxer: MediaMuxer? = null

    // ---- Camera2 fallback backend (video + sensors, no poses) --------------
    private var cameraManager: CameraManager? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var mediaRecorder: MediaRecorder? = null
    private var previewReader: ImageReader? = null
    /** True once the Camera2 fallback has been tried (only once, pre-start). */
    private var fallbackDone = false

    // ---- sensors -----------------------------------------------------------
    private var sensorManager: SensorManager? = null
    private val recordedSensors = mutableListOf<String>()

    // ---- writers -----------------------------------------------------------
    private var framesWriter: CsvWriter? = null
    private var posesWriter: CsvWriter? = null
    private var imuWriter: CsvWriter? = null
    private var magWriter: CsvWriter? = null
    private var lightWriter: CsvWriter? = null
    private var baroWriter: CsvWriter? = null

    private val watchdog = StreamWatchdog()

    private val flushRunnable = object : Runnable {
        override fun run() {
            if (!running) return
            flushAll()
            writerHandler.postDelayed(this, FLUSH_INTERVAL_MS)
        }
    }
    private val statusRunnable = object : Runnable {
        override fun run() {
            if (!running) return
            onStatus(currentStatus())
            statusHandler.postDelayed(this, STATUS_INTERVAL_MS)
        }
    }

    // Reusable line builders. Each is only ever touched on one thread.
    private val poseLine = StringBuilder(128)
    private val frameLine = StringBuilder(64)
    private val sensorLine = StringBuilder(128)

    // ------------------------------------------------------------------ start
    fun start() {
        if (running) return
        running = true
        captureDir.mkdirs()
        t0Nanos = SystemClock.elapsedRealtimeNanos()
        startWallclockMs = System.currentTimeMillis()

        // Writers first, so nothing is lost even if the camera is slow.
        framesWriter = CsvWriter(File(captureDir, "frames.csv"), "frame,t_s,file")
        posesWriter = CsvWriter(File(captureDir, "poses.csv"), "t_s,tx,ty,tz,qx,qy,qz,qw,tracking")
        imuWriter = CsvWriter(File(captureDir, "imu.csv"), "t_s,ax,ay,az,gx,gy,gz")
        magWriter = CsvWriter(File(captureDir, "mag.csv"), "t_s,mx,my,mz")
        lightWriter = CsvWriter(File(captureDir, "light.csv"), "t_s,lux")

        watchdog.setT0(t0Nanos)
        watchdog.setLogFile(File(captureDir, "watchdog.log"))
        watchdog.expect(StreamWatchdog.VIDEO_FRAMES)
        watchdog.expect(StreamWatchdog.POSES)
        watchdog.expect(StreamWatchdog.IMU)
        watchdog.expect(StreamWatchdog.MAGNETOMETER)
        watchdog.expect(StreamWatchdog.LIGHT)
        watchdog.stateDetailProvider = { stream -> stateDetail(stream) }
        watchdog.onStale = { stream, silentMs, detail ->
            staleStreams.add(stream)
            onWarning("$stream silent ${silentMs}ms — $detail")
        }
        watchdog.onRecovered = { stream ->
            staleStreams.remove(stream)
        }
        watchdog.start()

        // Sensors start immediately (independent of the camera warm-up).
        startSensors()

        // Capability manifest copy, so every capture carries its own probe.
        copyCapabilityManifest()

        // Camera + ARCore + video, on the recorder thread.
        recorderHandler.post { startRecordingBackend() }

        writerHandler.postDelayed(flushRunnable, FLUSH_INTERVAL_MS)
        statusHandler.postDelayed(statusRunnable, STATUS_INTERVAL_MS)
        writeMeta(final = false)
    }

    // ------------------------------------------------------------------ stop
    /** Stop everything and return the post-record self-check. Blocking. */
    fun stop(): StreamWatchdog.SelfCheck {
        if (!running) return watchdog.snapshot(0.0)
        running = false
        statusHandler.removeCallbacks(statusRunnable)
        recorderHandler.removeCallbacks(bringupWatchdogRunnable)
        writerHandler.removeCallbacks(flushRunnable)

        // 1. Stop the ARCore loop first, so no new poses arrive while we
        //    tear the session down. The loop thread finishes the encoder
        //    and muxer itself (drain + stop + release) before it exits.
        arcoreRunning = false
        try {
            arcoreThread?.join(5000)
        } catch (ignored: InterruptedException) {
        }
        arcoreThread = null
        // If the loop already released them, these are no-ops.
        try { mediaCodec?.stop() } catch (ignored: Exception) {}
        try { mediaCodec?.release() } catch (ignored: Exception) {}
        mediaCodec = null
        try { mediaMuxer?.stop() } catch (ignored: Exception) {}
        try { mediaMuxer?.release() } catch (ignored: Exception) {}
        mediaMuxer = null
        try { arSession?.pause() } catch (ignored: Exception) {}
        try { arSession?.close() } catch (ignored: Exception) {}
        arSession = null
        releaseEgl()

        // 2. If the Camera2 fallback ran, stop its video and close the camera.
        try {
            previewReader?.close()
        } catch (ignored: Exception) {
        }
        previewReader = null
        try {
            mediaRecorder?.stop()
        } catch (e: Exception) {
            Log.w(TAG, "MediaRecorder.stop failed", e)
        }
        try {
            mediaRecorder?.release()
        } catch (ignored: Exception) {
        }
        mediaRecorder = null
        try {
            captureSession?.close()
        } catch (ignored: Exception) {
        }
        captureSession = null
        try {
            cameraDevice?.close()
        } catch (ignored: Exception) {
        }
        cameraDevice = null

        // 3. Sensors off.
        try {
            sensorManager?.unregisterListener(sensorListener)
        } catch (ignored: Exception) {
        }

        watchdog.stop()
        flushAll()
        listOf(framesWriter, posesWriter, imuWriter, magWriter, lightWriter, baroWriter)
            .forEach { try { it?.close() } catch (ignored: Exception) { } }

        // 4. Self-check + final meta.
        val videoDurationS = lastFrameTs
        val check = watchdog.snapshot(videoDurationS)
        try {
            File(captureDir, "selfcheck.json").writeText(check.toJson().toString(2))
        } catch (e: Exception) {
            Log.w(TAG, "cannot write selfcheck.json", e)
        }
        writeMeta(final = true)

        recorderThread.quitSafely()
        statusThread.quitSafely()
        sensorThread.quitSafely()
        writerThread.quitSafely()
        return check
    }

    // ------------------------------------------------------- backend selection
    /**
     * Start recording. Main lens: ARCore owns the camera. If ARCore fails
     * to bring up, or produces zero frames/poses in the first 4 s, fall
     * back to plain Camera2 (video + sensors) — a PRE-START decision only.
     * Once ARCore is running with poses, it is never swapped away.
     */
    private fun startRecordingBackend() {
        if (lens == LensHelper.Lens.MAIN) {
            // ARCore owns the camera. startArCoreBackend only starts the
            // loop thread; a bringup failure is reported asynchronously
            // via onArCoreFailed, which falls back to Camera2 pre-start.
            startArCoreBackend()
            // Bring-up watchdog: ARCore must produce something soon, or it
            // is silently dead. Fires only before any pose exists, so it
            // cannot kill a working pose stream.
            recorderHandler.removeCallbacks(bringupWatchdogRunnable)
            recorderHandler.postDelayed(bringupWatchdogRunnable, BRINGUP_TIMEOUT_MS)
            return
        }
        backendNote = "ultrawide lens; Camera2 only (no ARCore poses)"
        if (!fallbackDone) {
            fallbackDone = true
            try {
                startCamera2Backend()
            } catch (e2: Exception) {
                Log.e(TAG, "Camera2 fallback failed", e2)
                onWarning("camera setup failed: ${e2.javaClass.simpleName}: ${e2.message}")
            }
        }
    }

    /** Pre-start only: if ARCore produced nothing in 4 s, it is dead. Tear
     *  it down and fall back to Camera2. Never fires after poses exist. */
    private val bringupWatchdogRunnable = object : Runnable {
        override fun run() {
            if (!running) return
            if (!arcoreActive) return
            if (poseCount > 0 || frameCount > 0) return
            if (fallbackDone) return
            Log.w(TAG, "ARCore produced nothing in 4s — falling back to Camera2")
            onWarning("ARCore 未出画面，改用普通相机录制（只有视频，没有路线）")
            fallbackDone = true
            backendNote = "arcore produced no frames in 4s; Camera2 fallback"
            try {
                // Signal the loop thread to stop, then pause the session
                // (that unblocks a session.update() stuck waiting for a
                // frame). Do NOT join here — the recorder thread must not
                // block. The loop thread cleans itself up.
                arcoreRunning = false
                try { arSession?.pause() } catch (ignored: Exception) {}
                teardownArCoreAttempt()
                startCamera2Backend()
            } catch (e: Exception) {
                Log.e(TAG, "Camera2 fallback failed", e)
                onWarning("camera setup failed: ${e.javaClass.simpleName}: ${e.message}")
            }
        }
    }

    /** Close everything an ARCore attempt opened, so a retry starts clean. */
    private fun teardownArCoreAttempt() {
        try { mediaCodec?.stop() } catch (ignored: Exception) {}
        try { mediaCodec?.release() } catch (ignored: Exception) {}
        mediaCodec = null
        try { mediaMuxer?.stop() } catch (ignored: Exception) {}
        try { mediaMuxer?.release() } catch (ignored: Exception) {}
        mediaMuxer = null
        try { arSession?.pause() } catch (ignored: Exception) {}
        try { arSession?.close() } catch (ignored: Exception) {}
        arSession = null
        releaseEgl()
        arcoreActive = false
        arcoreRunning = false
    }

    // ---------------------------------------------------------- ARCore backend
    /**
     * Start ARCore as the camera owner. This only starts the loop thread;
     * the thread itself does EGL + Session + configure + resume + the loop
     * (reverse-engineered from the old app). EGL MUST be created on the
     * same thread that calls session.update() — in v2.1.0 EGL was created
     * on the recorder thread while update() ran on the loop thread, so
     * ARCore produced nothing and the preview stayed black.
     * A bringup failure is reported back to the recorder thread, which
     * falls back to Camera2 (pre-start only).
     */
    private fun startArCoreBackend() {
        arcoreRunning = true
        arcoreThread = Thread({ arcoreLoopEntry() }, "arcore-loop").apply { start() }
    }

    /** ARCore loop thread entry: bring up EGL + Session, then run the loop.
     *  Owns the session and EGL; cleans them up when it exits. */
    private fun arcoreLoopEntry() {
        var session: Session? = null
        try {
            initEglTexture()
            session = Session(context)
            session.setCameraTextureName(cameraTexId)
            val config = Config(session)
            config.updateMode = Config.UpdateMode.BLOCKING
            config.focusMode = Config.FocusMode.AUTO
            config.planeFindingMode = Config.PlaneFindingMode.DISABLED
            config.lightEstimationMode = Config.LightEstimationMode.DISABLED
            config.depthMode = Config.DepthMode.DISABLED
            // Camera config: 30 fps, capped at 720p. The old app took the
            // largest config (1080p) and its software encoder fell below 8 fps.
            val depthSupported = session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
            val chosen = pickCameraConfig(session, depthSupported)
                ?: throw IllegalStateException("no ARCore camera config")
            session.cameraConfig = chosen
            session.configure(config)
            session.resume()
            arSession = session
            videoSize = chosen.imageSize
            cameraId = chosen.cameraId
            arcoreActive = true
            trackingState = "PAUSED"
            backendNote = "arcore (owns camera), ${videoSize.width}x${videoSize.height} 30fps"
            Log.i(TAG, "ARCore up: ${videoSize.width}x${videoSize.height}, camera=$cameraId")
            runArCoreLoop(session)
        } catch (e: Exception) {
            Log.e(TAG, "ARCore backend failed", e)
            // Report on the recorder thread (it owns the fallback).
            recorderHandler.post { onArCoreFailed(e) }
        } finally {
            // The loop releases codec/muxer itself; here we release the
            // session + EGL. stop() also tries these (guarded), so a
            // double release is harmless.
            try { session?.pause() } catch (ignored: Exception) {}
            try { session?.close() } catch (ignored: Exception) {}
            if (arSession === session) arSession = null
            releaseEgl()
            arcoreActive = false
            arcoreRunning = false
        }
    }

    /** ARCore failed. Pre-start (no poses yet): fall back to Camera2.
     *  Mid-capture (poses exist): NEVER swap — that was the old bug.
     *  Just report it and keep the poses we have. */
    private fun onArCoreFailed(e: Exception) {
        if (!running) return
        if (fallbackDone) return
        if (poseCount > 0 || frameCount > 0) {
            backendNote = "arcore failed mid-capture (${e.javaClass.simpleName}); kept $poseCount poses"
            onWarning("ARCore 中途停止，已保留 $poseCount 个路线点")
            return
        }
        fallbackDone = true
        backendNote = "arcore bring-up failed (${e.javaClass.simpleName}: ${e.message}); Camera2 fallback"
        onWarning("ARCore 启动失败，改用普通相机录制（只有视频，没有路线）")
        try {
            teardownArCoreAttempt()
            startCamera2Backend()
        } catch (e2: Exception) {
            Log.e(TAG, "Camera2 fallback failed", e2)
            onWarning("camera setup failed: ${e2.javaClass.simpleName}: ${e2.message}")
        }
    }

    /** Pick the best ARCore camera config: 30 fps preferred, size capped at
     *  720p (larger only costs encoder time). Falls back to any config. */
    private fun pickCameraConfig(session: Session, depthSupported: Boolean): CameraConfig? {
        val rungs: List<(CameraConfigFilter) -> Unit> = listOf(
            { f ->
                f.targetFps = EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30)
                if (depthSupported) f.depthSensorUsage =
                    EnumSet.of(CameraConfig.DepthSensorUsage.REQUIRE_AND_USE)
            },
            { f -> f.targetFps = EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30) },
            { },
        )
        for (apply in rungs) {
            val filter = CameraConfigFilter(session)
            apply(filter)
            val list = session.getSupportedCameraConfigs(filter)
            if (list.isNotEmpty()) {
                // Largest config that still fits in 720p; if none fits, the
                // smallest config overall (a slow phone needs the small one).
                val capped = list.filter {
                    it.imageSize.width <= MAX_VIDEO_WIDTH && it.imageSize.height <= MAX_VIDEO_HEIGHT
                }
                val pool = if (capped.isNotEmpty()) capped else list
                return pool.maxByOrNull { it.imageSize.width * it.imageSize.height }
            }
        }
        return null
    }

    /** EGL display + GLES2 context + 1x1 pbuffer + a GL texture id, handed
     *  to ARCore via setCameraTextureName. Must run on the thread that will
     *  call session.update() (the ARCore loop thread owns it). */
    private fun initEglTexture() {
        val display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (display == EGL14.EGL_NO_DISPLAY) throw IllegalStateException("no EGL display")
        val version = IntArray(2)
        if (!EGL14.eglInitialize(display, version, 0, version, 1)) {
            throw IllegalStateException("eglInitialize failed")
        }
        val configAttribs = intArrayOf(
            EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8, EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES_BIT,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT, EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        if (!EGL14.eglChooseConfig(display, configAttribs, 0, configs, 0, 1, numConfigs, 0)
            || numConfigs[0] <= 0
        ) {
            throw IllegalStateException("eglChooseConfig failed")
        }
        val ctxAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
        val ctx = EGL14.eglCreateContext(display, configs[0], EGL14.EGL_NO_CONTEXT, ctxAttribs, 0)
        if (ctx == EGL14.EGL_NO_CONTEXT) throw IllegalStateException("eglCreateContext failed")
        val surfAttribs = intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE)
        val surf = EGL14.eglCreatePbufferSurface(display, configs[0], surfAttribs, 0)
        if (surf == EGL14.EGL_NO_SURFACE) throw IllegalStateException("eglCreatePbufferSurface failed")
        if (!EGL14.eglMakeCurrent(display, surf, surf, ctx)) {
            throw IllegalStateException("eglMakeCurrent failed")
        }
        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        if (tex[0] == 0) throw IllegalStateException("glGenTextures failed")
        eglDisplay = display
        eglContext = ctx
        eglSurface = surf
        cameraTexId = tex[0]
    }

    private fun releaseEgl() {
        val d = eglDisplay ?: return
        try {
            EGL14.eglMakeCurrent(d, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        } catch (ignored: Exception) {
        }
        try {
            eglSurface?.let { EGL14.eglDestroySurface(d, it) }
        } catch (ignored: Exception) {
        }
        try {
            eglContext?.let { EGL14.eglDestroyContext(d, it) }
        } catch (ignored: Exception) {
        }
        try {
            EGL14.eglTerminate(d)
        } catch (ignored: Exception) {
        }
        eglDisplay = null
        eglContext = null
        eglSurface = null
        cameraTexId = 0
    }

    /** The ARCore loop body. Runs on the arcore-loop thread (which also did
     *  the EGL + Session bringup, so the GL context is current here). Per
     *  frame:
     *  1. pose — written UNCONDITIONALLY, before the encoder is touched
     *     (the old app gated poses on the encoder; that is the bug);
     *  2. video — acquire the camera image, bulk-convert YUV->NV12, feed
     *     MediaCodec; if the encoder is busy the video frame is skipped
     *     but the pose is already written;
     *  3. preview thumbnail every 6th encoded frame. */
    private fun runArCoreLoop(session: Session) {
        val w = videoSize.width
        val h = videoSize.height
        var codec: MediaCodec? = null
        var muxer: MediaMuxer? = null
        val muxerState = MuxerState()
        val bufferInfo = MediaCodec.BufferInfo()
        // Reusable NV12 buffer + row scratch, so the loop allocates
        // nothing per frame.
        val nv12 = ByteArray(w * h * 3 / 2)
        val yRow = ByteArray(w)
        val uvRowU = ByteArray(w / 2)
        val uvRowV = ByteArray(w / 2)
        val thumbPixels = IntArray((PREVIEW_MAX_DIM * PREVIEW_MAX_DIM * 9) / 16 + 1024)
        var encodedFrames = 0
        var t0Ns: Long? = null
        try {
                codec = MediaCodec.createEncoderByType("video/avc")
                val format = MediaFormat.createVideoFormat("video/avc", w, h)
                format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                format.setInteger(MediaFormat.KEY_BIT_RATE, VIDEO_BITRATE)
                format.setInteger(MediaFormat.KEY_FRAME_RATE, VIDEO_FPS)
                format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
                codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                muxer = MediaMuxer(File(captureDir, "video.mp4").absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
                muxer.setOrientationHint(0)
                mediaCodec = codec
                mediaMuxer = muxer
                codec.start()
                while (arcoreRunning) {
                    val frame: Frame
                    try {
                        frame = session.update()
                    } catch (e: Exception) {
                        if (arcoreRunning) Log.w(TAG, "ARCore update error (continuing)", e)
                        continue
                    }
                    val ts = frame.timestamp
                    if (ts == 0L) continue
                    if (t0Ns == null) t0Ns = ts
                    val t = (ts - t0Nanos) / 1e9

                    // --- 1. POSE: unconditional, first. ---
                    val camera = frame.camera
                    val pose: Pose = camera.pose
                    val tracking: TrackingState = camera.trackingState
                    trackingState = tracking.name
                    poseLine.setLength(0)
                    poseLine.append(t).append(',')
                        .append(pose.tx()).append(',')
                        .append(pose.ty()).append(',')
                        .append(pose.tz()).append(',')
                        .append(pose.qx()).append(',')
                        .append(pose.qy()).append(',')
                        .append(pose.qz()).append(',')
                        .append(pose.qw()).append(',')
                        .append(tracking.name).append('\n')
                    posesWriter?.append(poseLine)
                    poseCount++
                    watchdog.noteSample(StreamWatchdog.POSES, ts)

                    // Intrinsics once (for meta.json / reconstruction).
                    if (intrinsicsJson == null) {
                        try {
                            val intr: CameraIntrinsics = camera.imageIntrinsics
                            intrinsicsJson = JSONObject()
                                .put("source", "arcore-image-intrinsics")
                                .put("width", w).put("height", h)
                                .put("fx_px", intr.focalLength[0])
                                .put("fy_px", intr.focalLength[1])
                                .put("cx_px", intr.principalPoint[0])
                                .put("cy_px", intr.principalPoint[1])
                        } catch (ignored: Exception) {
                        }
                    }

                    // --- 2. VIDEO: acquire image, convert, encode. ---
                    val image: Image
                    try {
                        image = frame.acquireCameraImage()
                    } catch (e: NotYetAvailableException) {
                        continue
                    } catch (e: Exception) {
                        if (arcoreRunning) Log.w(TAG, "acquireCameraImage failed", e)
                        continue
                    }
                    var queued = false
                    try {
                        val inIdx = codec.dequeueInputBuffer(10_000)
                        if (inIdx >= 0) {
                            val inBuf = codec.getInputBuffer(inIdx)
                            if (inBuf != null) {
                                yuv420ToNv12(image, nv12, yRow, uvRowU, uvRowV)
                                inBuf.clear()
                                inBuf.put(nv12, 0, minOf(nv12.size, inBuf.remaining()))
                                codec.queueInputBuffer(
                                    inIdx, 0, minOf(nv12.size, inBuf.capacity()),
                                    (ts - t0Ns!!) / 1000, 0
                                )
                                queued = true
                            }
                        }
                    } catch (e: Exception) {
                        if (arcoreRunning) Log.w(TAG, "video encode failed for a frame", e)
                    }
                    if (queued) {
                        encodedFrames++
                        frameCount++
                        lastFrameTs = t
                        frameLine.setLength(0)
                        frameLine.append(frameCount).append(',').append(t).append(",video.mp4\n")
                        framesWriter?.append(frameLine)
                        watchdog.noteSample(StreamWatchdog.VIDEO_FRAMES, ts)
                        if (encodedFrames % PREVIEW_EVERY_N_FRAMES == 0) {
                            try {
                                yuvThumbnail(image, thumbPixels)?.let { onPreview(it) }
                            } catch (ignored: Exception) {
                            }
                        }
                    }
                    try {
                        image.close()
                    } catch (ignored: Exception) {
                    }
                    drain(codec, bufferInfo, muxer, muxerState, false)
                }
            } catch (e: Exception) {
                Log.e(TAG, "ARCore loop failed", e)
                onWarning("ARCore 录制中断：${e.javaClass.simpleName}: ${e.message}")
            } finally {
                // Finish the encoder and muxer so video.mp4 is playable.
                try {
                    codec?.let {
                        try { it.signalEndOfInputStream() } catch (ignored: Exception) {}
                        drain(it, bufferInfo, muxer, muxerState, true)
                        try { it.stop() } catch (ignored: Exception) {}
                        it.release()
                    }
                } catch (ignored: Exception) {
                }
                mediaCodec = null
                try {
                    if (muxerState.started) {
                        try { muxer?.stop() } catch (ignored: Exception) {}
                    }
                    muxer?.release()
                } catch (ignored: Exception) {
                }
                mediaMuxer = null
            }
    }

    /** Small muxer state holder for drain(). */
    private class MuxerState {
        var track: Int = -1
        var started: Boolean = false
    }

    /** Pull encoded output from the codec into the muxer. */
    private fun drain(
        codec: MediaCodec,
        bufferInfo: MediaCodec.BufferInfo,
        muxer: MediaMuxer?,
        state: MuxerState,
        endOfStream: Boolean,
    ) {
        if (muxer == null) return
        while (true) {
            val outIdx = try {
                codec.dequeueOutputBuffer(bufferInfo, 0)
            } catch (e: Exception) {
                return
            }
            if (outIdx == MediaCodec.INFO_TRY_AGAIN_LATER) return
            if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (state.started) return
                val format = codec.outputFormat
                state.track = muxer.addTrack(format)
                muxer.start()
                state.started = true
            } else if (outIdx >= 0) {
                val outBuf = codec.getOutputBuffer(outIdx)
                if (outBuf != null && bufferInfo.size > 0 && state.started) {
                    outBuf.position(bufferInfo.offset)
                    outBuf.limit(bufferInfo.offset + bufferInfo.size)
                    try {
                        muxer.writeSampleData(state.track, outBuf, bufferInfo)
                    } catch (ignored: Exception) {
                    }
                }
                try {
                    codec.releaseOutputBuffer(outIdx, false)
                } catch (ignored: Exception) {
                }
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
            }
        }
    }

    /**
     * YUV_420_888 -> NV12, bulk row copies. The old app did per-pixel
     * ByteBuffer.get() (~3M calls per 1080p frame) on the same thread as
     * session.update(); this version copies whole rows, which is what
     * keeps the loop at 30 fps.
     */
    private fun yuv420ToNv12(
        image: Image,
        out: ByteArray,
        yRow: ByteArray,
        uvRowU: ByteArray,
        uvRowV: ByteArray,
    ) {
        val w = image.width
        val h = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        var pos = 0
        val yBuf = yPlane.buffer
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        if (yPixelStride == 1) {
            for (r in 0 until h) {
                yBuf.position(r * yRowStride)
                yBuf.get(yRow, 0, w)
                System.arraycopy(yRow, 0, out, pos, w)
                pos += w
            }
        } else {
            for (r in 0 until h) {
                for (c in 0 until w) {
                    out[pos++] = yBuf.get(r * yRowStride + c * yPixelStride)
                }
            }
        }
        val uvH = h / 2
        val uvW = w / 2
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val uRowStride = uPlane.rowStride
        val vRowStride = vPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vPixelStride = vPlane.pixelStride
        for (r in 0 until uvH) {
            if (uPixelStride == 1) {
                uBuf.position(r * uRowStride)
                uBuf.get(uvRowU, 0, uvW)
            } else {
                for (c in 0 until uvW) uvRowU[c] = uBuf.get(r * uRowStride + c * uPixelStride)
            }
            if (vPixelStride == 1) {
                vBuf.position(r * vRowStride)
                vBuf.get(uvRowV, 0, uvW)
            } else {
                for (c in 0 until uvW) uvRowV[c] = vBuf.get(r * vRowStride + c * vPixelStride)
            }
            for (c in 0 until uvW) {
                out[pos++] = uvRowU[c]
                out[pos++] = uvRowV[c]
            }
        }
    }

    /**
     * Small grayscale preview thumbnail from the Y plane, downsampled.
     * Grayscale is enough to see the camera is alive and what it points
     * at; color would cost a second full-frame conversion per thumbnail.
     */
    private fun yuvThumbnail(image: Image, pixels: IntArray): Bitmap? {
        val w = image.width
        val h = image.height
        val scale = maxOf(1, maxOf(w, h) / PREVIEW_MAX_DIM)
        val tw = w / scale
        val th = h / scale
        if (tw * th > pixels.size) return null
        val yPlane = image.planes[0]
        val yBuf = yPlane.buffer
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        var p = 0
        for (r in 0 until th) {
            val srcRow = r * scale
            for (c in 0 until tw) {
                val srcCol = c * scale
                val y = yBuf.get(srcRow * yRowStride + srcCol * yPixelStride).toInt() and 0xFF
                pixels[p++] = 0xFF000000.toInt() or (y shl 16) or (y shl 8) or y
            }
        }
        val bmp = Bitmap.createBitmap(tw, th, Bitmap.Config.ARGB_8888)
        bmp.setPixels(pixels, 0, tw, 0, 0, tw, th)
        return bmp
    }

    // ------------------------------------------- Camera2 fallback (no poses)
    /** Plain Camera2 + MediaRecorder. Only used when ARCore cannot run
     *  (ultrawide lens, or ARCore failed before any pose existed). Records
     *  video + sensors; no route. This path is the old app's Camera2
     *  backend, which is fast (hardware encoding) and correct. */
    private fun startCamera2Backend() {
        cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val lenses = LensHelper.enumerateBackCameras(context)
        lensInfo = lenses.firstOrNull { it.lens == lens }
            ?: lenses.firstOrNull()
        val info = lensInfo
        if (info == null) {
            onWarning("no back camera found — recording sensors only")
            return
        }
        val sizes = info.videoSizes
        videoSize = sizes.firstOrNull { it.width == 1920 && it.height == 1080 }
            ?: sizes.maxByOrNull { it.width * it.height }
            ?: Size(1920, 1080)
        cameraId = info.cameraId
        arcoreActive = false
        trackingState = if (lens == LensHelper.Lens.ULTRAWIDE) {
            "N/A (ultrawide)"
        } else {
            "N/A (Camera2 only)"
        }
        try {
            mediaRecorder = createMediaRecorder()
        } catch (e: Exception) {
            Log.e(TAG, "MediaRecorder setup failed", e)
            onWarning("video setup failed: ${e.message}")
            return
        }
        openCamera()
        writeMeta(final = false)
    }

    private fun createMediaRecorder(): MediaRecorder {
        val mr = if (Build.VERSION.SDK_INT >= 31) MediaRecorder(context)
        else @Suppress("DEPRECATION") MediaRecorder()
        mr.setVideoSource(MediaRecorder.VideoSource.SURFACE)
        mr.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        mr.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
        mr.setVideoSize(videoSize.width, videoSize.height)
        mr.setVideoFrameRate(VIDEO_FPS)
        mr.setVideoEncodingBitRate(20_000_000)
        mr.setOutputFile(File(captureDir, "video.mp4").absolutePath)
        mr.prepare()
        return mr
    }

    private fun openCamera() {
        val manager = cameraManager ?: return
        val deviceCallback = object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                cameraDevice = device
                createCaptureSession()
            }

            override fun onDisconnected(device: CameraDevice) {
                onWarning("camera disconnected")
            }

            override fun onError(device: CameraDevice, code: Int) {
                onWarning("camera error code=$code")
            }
        }
        try {
            manager.openCamera(cameraId, deviceCallback, recorderHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "openCamera failed", e)
            onWarning("camera open failed: ${e.message}")
        } catch (e: SecurityException) {
            Log.e(TAG, "openCamera failed", e)
            onWarning("camera permission missing")
        } catch (e: Exception) {
            Log.e(TAG, "openCamera failed", e)
            onWarning("camera open failed: ${e.javaClass.simpleName}: ${e.message}")
        }
    }

    @Suppress("DEPRECATION")
    private fun createCaptureSession() {
        val device = cameraDevice ?: return
        val surfaces = mutableListOf<Surface>()
        surfaces.add(mediaRecorder!!.surface)
        // Preview ImageReader (small, YUV) so the preview works in Camera2
        // mode too. Without it the preview stays black whenever the app
        // falls back to Camera2 — that is what v2.1.0 showed.
        try {
            previewReader?.close()
        } catch (ignored: Exception) {
        }
        previewReader = null
        try {
            val reader = ImageReader.newInstance(640, 360, ImageFormat.YUV_420_888, 2)
            val previewPixels = IntArray(640 * 360)
            var previewCounter = 0
            reader.setOnImageAvailableListener({ r ->
                val image = try {
                    r.acquireLatestImage()
                } catch (ignored: Exception) {
                    null
                } ?: return@setOnImageAvailableListener
                try {
                    // Throttle: one thumbnail every ~6th preview frame.
                    previewCounter++
                    if (previewCounter % 6 == 0) {
                        yuvThumbnail(image, previewPixels)?.let { onPreview(it) }
                    }
                } catch (ignored: Exception) {
                } finally {
                    try {
                        image.close()
                    } catch (ignored: Exception) {
                    }
                }
            }, recorderHandler)
            surfaces.add(reader.surface)
            previewReader = reader
        } catch (e: Exception) {
            Log.w(TAG, "preview ImageReader failed (recording continues)", e)
        }
        val builder: CaptureRequest.Builder
        try {
            builder = device.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "createCaptureRequest failed", e)
            onWarning("capture request failed: ${e.message}")
            return
        }
        for (s in surfaces) builder.addTarget(s)
        builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
        builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
        try {
            val chars = cameraManager!!.getCameraCharacteristics(cameraId)
            val stab = chars.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)
            if (stab?.contains(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON) == true) {
                builder.set(
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
                )
            }
        } catch (ignored: Exception) {
        }
        val request = builder.build()
        val sessionCallback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                captureSession = session
                try {
                    session.setRepeatingRequest(request, captureCallback, recorderHandler)
                } catch (e: CameraAccessException) {
                    Log.e(TAG, "setRepeatingRequest failed", e)
                    onWarning("repeating request failed: ${e.message}")
                    return
                }
                try {
                    mediaRecorder?.start()
                } catch (e: Exception) {
                    Log.e(TAG, "MediaRecorder.start failed", e)
                    onWarning("video start failed: ${e.message}")
                }
                writeMeta(final = false)
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {
                Log.e(TAG, "camera capture session configuration failed")
                onWarning("camera capture session configuration failed")
            }
        }
        try {
            device.createCaptureSession(surfaces, sessionCallback, recorderHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "createCaptureSession failed", e)
            onWarning("capture session failed: ${e.message}")
        } catch (e: Exception) {
            Log.e(TAG, "createCaptureSession failed", e)
            onWarning("capture session failed: ${e.javaClass.simpleName}: ${e.message}")
        }
    }

    /** One row per captured video frame (Camera2 fallback only). Runs on the
     *  recorder thread. The ARCore backend writes frames.csv from its loop. */
    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult,
        ) {
            val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
            frameCount++
            val t = (ts - t0Nanos) / 1e9
            lastFrameTs = t
            frameLine.setLength(0)
            frameLine.append(frameCount).append(',').append(t).append(",video.mp4\n")
            framesWriter?.append(frameLine)
            watchdog.noteSample(StreamWatchdog.VIDEO_FRAMES, ts)
        }
    }

    // --------------------------------------------------------------- sensors
    private val latestAccel = FloatArray(3)
    private val latestGyro = FloatArray(3)

    private val sensorListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            val t = (event.timestamp - t0Nanos) / 1e9
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    latestAccel[0] = event.values[0]
                    latestAccel[1] = event.values[1]
                    latestAccel[2] = event.values[2]
                    // One IMU row per accelerometer event, carrying the
                    // latest gyroscope sample. Accel and gyro come from the
                    // same chip at the same rate, so nothing is lost.
                    sensorLine.setLength(0)
                    sensorLine.append(t).append(',')
                        .append(latestAccel[0]).append(',')
                        .append(latestAccel[1]).append(',')
                        .append(latestAccel[2]).append(',')
                        .append(latestGyro[0]).append(',')
                        .append(latestGyro[1]).append(',')
                        .append(latestGyro[2]).append('\n')
                    imuWriter?.append(sensorLine)
                    watchdog.noteSample(StreamWatchdog.IMU, event.timestamp)
                }
                Sensor.TYPE_GYROSCOPE -> {
                    latestGyro[0] = event.values[0]
                    latestGyro[1] = event.values[1]
                    latestGyro[2] = event.values[2]
                }
                Sensor.TYPE_MAGNETIC_FIELD -> {
                    sensorLine.setLength(0)
                    sensorLine.append(t).append(',')
                        .append(event.values[0]).append(',')
                        .append(event.values[1]).append(',')
                        .append(event.values[2]).append('\n')
                    magWriter?.append(sensorLine)
                    watchdog.noteSample(StreamWatchdog.MAGNETOMETER, event.timestamp)
                }
                Sensor.TYPE_LIGHT -> {
                    sensorLine.setLength(0)
                    sensorLine.append(t).append(',').append(event.values[0]).append('\n')
                    lightWriter?.append(sensorLine)
                    watchdog.noteSample(StreamWatchdog.LIGHT, event.timestamp)
                }
                Sensor.TYPE_PRESSURE -> {
                    sensorLine.setLength(0)
                    sensorLine.append(t).append(',').append(event.values[0]).append('\n')
                    baroWriter?.append(sensorLine)
                    watchdog.noteSample(StreamWatchdog.BAROMETER, event.timestamp)
                }
            }
        }

        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
    }

    private fun startSensors() {
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sensorManager = sm
        // Accelerometer + gyroscope: fastest (the old app reached ~200 Hz).
        sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let {
            sm.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_FASTEST, sensorHandler)
            recordedSensors.add("accelerometer")
        }
        sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let {
            sm.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_FASTEST, sensorHandler)
            recordedSensors.add("gyroscope")
        }
        sm.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)?.let {
            sm.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_GAME, sensorHandler)
            recordedSensors.add("magnetometer")
        }
        sm.getDefaultSensor(Sensor.TYPE_LIGHT)?.let {
            sm.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_NORMAL, sensorHandler)
            recordedSensors.add("light")
        }
        // Barometer: auto-detect. Never assumed; recorded only if present.
        sm.getDefaultSensor(Sensor.TYPE_PRESSURE)?.let {
            baroWriter = CsvWriter(File(captureDir, "baro.csv"), "t_s,hPa")
            sm.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_NORMAL, sensorHandler)
            recordedSensors.add("barometer")
            watchdog.expect(StreamWatchdog.BAROMETER)
        }
    }

    // ------------------------------------------------------------ housekeeping
    private fun flushAll() {
        framesWriter?.flush()
        posesWriter?.flush()
        imuWriter?.flush()
        magWriter?.flush()
        lightWriter?.flush()
        baroWriter?.flush()
    }

    private fun currentStatus(): Status {
        val elapsed = if (t0Nanos > 0) (SystemClock.elapsedRealtimeNanos() - t0Nanos) / 1e9 else 0.0
        return Status(
            elapsedS = elapsed,
            frameCount = frameCount,
            poseCount = poseCount,
            tracking = trackingState,
            lens = lens,
            arcoreActive = arcoreActive,
            staleStreams = staleStreams.toList(),
            backend = if (arcoreActive) "ARCore" else "普通相机",
        )
    }

    /** Short state line per stream, read by the watchdog when a stream stalls. */
    private fun stateDetail(stream: String): String = when (stream) {
        StreamWatchdog.POSES -> "arcore_active=$arcoreActive tracking=$trackingState loop_running=$arcoreRunning"
        StreamWatchdog.VIDEO_FRAMES -> "arcore_active=$arcoreActive camera2_open=${cameraDevice != null} frames=$frameCount"
        StreamWatchdog.IMU, StreamWatchdog.MAGNETOMETER, StreamWatchdog.LIGHT, StreamWatchdog.BAROMETER ->
            "sensor_listener_registered=${sensorManager != null}"
        else -> ""
    }

    private fun copyCapabilityManifest() {
        try {
            val src = File(context.filesDir, "capability.json")
            if (src.exists()) {
                src.copyTo(File(captureDir, "capability.json"), overwrite = true)
            }
        } catch (ignored: Exception) {
        }
    }

    /** Write meta.json. Called at start (so a crash still leaves one) and
     *  again at stop with the final counts. */
    private fun writeMeta(final: Boolean) {
        try {
            val info = lensInfo
            val root = JSONObject()
            root.put("device_model", android.os.Build.MODEL ?: "unknown")
            root.put("manufacturer", android.os.Build.MANUFACTURER ?: "unknown")
            root.put("android_version", android.os.Build.VERSION.RELEASE ?: "unknown")
            root.put("app_version", "2.1.1")
            root.put("lens", lens.name.lowercase())
            root.put("lens_camera_id", cameraId)
            root.put("lens_focal_length_mm", info?.focalLengthMm?.toDouble() ?: JSONObject.NULL)
            root.put("lens_fov_deg", info?.hFovDeg ?: JSONObject.NULL)
            val video = JSONObject()
            video.put("width", videoSize.width)
            video.put("height", videoSize.height)
            video.put("fps", VIDEO_FPS)
            video.put("codec", "h264")
            video.put("bitrate_bps", VIDEO_BITRATE)
            root.put("video", video)
            root.put("sensors_recorded", JSONArray(recordedSensors))
            root.put("arcore_poses", arcoreActive)
            if (!arcoreActive) {
                root.put(
                    "arcore_poses_reason",
                    if (lens == LensHelper.Lens.ULTRAWIDE) "ultrawide lens selected; ARCore requires the main camera"
                    else "ARCore unavailable: $trackingState"
                )
            }
            root.put("backend_note", backendNote)
            intrinsicsJson?.let { root.put("camera_intrinsics", it) }
            root.put("start_wallclock_ms", startWallclockMs)
            root.put("t0_elapsed_realtime_nanos", t0Nanos)
            if (final) {
                root.put("duration_s", lastFrameTs)
                root.put("frame_count", frameCount)
                root.put("pose_count", poseCount)
            }
            File(captureDir, "meta.json").writeText(root.toString(2))
        } catch (e: Exception) {
            Log.w(TAG, "cannot write meta.json", e)
        }
    }
}
