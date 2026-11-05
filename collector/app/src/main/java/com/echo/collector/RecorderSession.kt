package com.echo.collector

import android.content.Context
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
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Surface
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.UnavailableException
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Collections
import java.util.EnumSet

/**
 * One recording session. Owns the camera, the ARCore session, the sensors,
 * every CSV writer, and the stream watchdog. Lives inside
 * [RecorderService] (a foreground service) so it survives the screen
 * turning off and Android does not kill it mid-walk.
 *
 * THE POSE BUG, AND HOW THIS CLASS FIXES IT
 * ------------------------------------------
 * The old app's ARCore pose stream died after ~6 s. The analysis of the
 * 2026-11-05 capture showed ARCore delivered "10 frames in 6 s" and was
 * then cut over to Camera2-only; poses never resumed. That happens when
 * the ARCore update loop is starved or the session is paused and nobody
 * resumes it. This class removes every cause:
 * - The ARCore session is created ONCE at start and resumed ONCE, and is
 *   paused only in stop(). There is no watchdog, no fallback, and no
 *   mode switch that can pause or replace it mid-recording.
 * - session.update() runs in a tight loop on its OWN dedicated thread
 *   ("arcore-update"). That thread does exactly one thing: copy the pose
 *   (7 floats + timestamp + tracking state) into a buffer. It never does
 *   I/O, never touches the UI, and never blocks on a lock held by anyone
 *   else, so it cannot be starved.
 * - The whole session lives in a foreground service with a partial wake
 *   lock, so Android cannot throttle or kill the process when the screen
 *   goes off or the app is backgrounded.
 * - If tracking is lost, we keep recording and log the tracking state.
 *   Losing tracking is data (a PAUSED row in poses.csv), not a reason to
 *   stop. Nothing ever stops the pose stream except the user pressing Stop.
 *
 * LENSES
 * ------
 * Main lens: ARCore shared camera. Video + poses + sensors.
 * Ultrawide lens: Camera2 only (video + sensors). ARCore's shared camera
 * only supports the camera ARCore itself chooses (the main); the
 * ultrawide cannot be used for tracking. This is reported honestly in
 * the UI and in meta.json (arcore_poses=false) instead of silently
 * producing an empty poses.csv.
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
) {

    companion object {
        private const val TAG = "RecorderSession"
        private const val VIDEO_WIDTH = 1920
        private const val VIDEO_HEIGHT = 1080
        private const val VIDEO_FPS = 30
        private const val VIDEO_BITRATE = 20_000_000
        private const val STATUS_INTERVAL_MS = 500L
        private const val FLUSH_INTERVAL_MS = 1000L
    }

    data class Status(
        val elapsedS: Double,
        val frameCount: Long,
        val poseCount: Long,
        val tracking: String,
        val lens: LensHelper.Lens,
        val arcoreActive: Boolean,
        val staleStreams: List<String>,
    )

    // ---- threads -----------------------------------------------------------
    private val recorderThread = HandlerThread("recorder").apply { start() }
    private val recorderHandler = Handler(recorderThread.looper)
    private val sensorThread = HandlerThread("sensors").apply { start() }
    private val sensorHandler = Handler(sensorThread.looper)
    private val writerThread = HandlerThread("writer").apply { start() }
    private val writerHandler = Handler(writerThread.looper)

    // ---- session state -----------------------------------------------------
    @Volatile private var running = false
    @Volatile private var updateRunning = false
    private var t0Nanos: Long = -1
    private var startWallclockMs: Long = -1
    private var cameraId: String = ""
    private var lensInfo: LensHelper.LensInfo? = null
    private var videoSize = Size(VIDEO_WIDTH, VIDEO_HEIGHT)
    private var arcoreActive = false
    @Volatile private var trackingState: String = "UNKNOWN"
    @Volatile private var poseCount: Long = 0
    @Volatile private var frameCount: Long = 0
    @Volatile private var lastFrameTs: Double = 0.0
    private val staleStreams = Collections.synchronizedSet(mutableSetOf<String>())

    // ---- camera / ARCore ---------------------------------------------------
    private var cameraManager: CameraManager? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var mediaRecorder: MediaRecorder? = null
    private var arSession: Session? = null
    private var sharedCamera: com.google.ar.core.SharedCamera? = null
    private var updateThread: Thread? = null

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
            recorderHandler.postDelayed(this, STATUS_INTERVAL_MS)
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

        // Camera + ARCore + video, on the recorder thread (Camera2 is async).
        recorderHandler.post { startCameraAndArCore() }

        writerHandler.postDelayed(flushRunnable, FLUSH_INTERVAL_MS)
        recorderHandler.postDelayed(statusRunnable, STATUS_INTERVAL_MS)
        writeMeta(final = false)
    }

    // ------------------------------------------------------------------ stop
    /** Stop everything and return the post-record self-check. Blocking. */
    fun stop(): StreamWatchdog.SelfCheck {
        if (!running) return watchdog.snapshot(0.0)
        running = false
        recorderHandler.removeCallbacks(statusRunnable)
        recorderHandler.removeCallbacks(cameraWatchdogRunnable)
        writerHandler.removeCallbacks(flushRunnable)

        // 1. Stop the ARCore update loop first, so no new poses arrive
        //    while we tear the session down.
        updateRunning = false
        try {
            arSession?.pause()
        } catch (ignored: Exception) {
        }
        try {
            updateThread?.join(2000)
        } catch (ignored: InterruptedException) {
        }
        updateThread = null

        // 2. Stop the video. stop() can throw if nothing was recorded.
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

        // 3. Close camera.
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

        // 4. Close ARCore.
        try {
            arSession?.close()
        } catch (ignored: Exception) {
        }
        arSession = null
        sharedCamera = null

        // 5. Sensors off.
        try {
            sensorManager?.unregisterListener(sensorListener)
        } catch (ignored: Exception) {
        }

        watchdog.stop()
        flushAll()
        listOf(framesWriter, posesWriter, imuWriter, magWriter, lightWriter, baroWriter)
            .forEach { try { it?.close() } catch (ignored: Exception) { } }

        // 6. Self-check + final meta.
        val videoDurationS = lastFrameTs
        val check = watchdog.snapshot(videoDurationS)
        try {
            File(captureDir, "selfcheck.json").writeText(check.toJson().toString(2))
        } catch (e: Exception) {
            Log.w(TAG, "cannot write selfcheck.json", e)
        }
        writeMeta(final = true)

        recorderThread.quitSafely()
        sensorThread.quitSafely()
        writerThread.quitSafely()
        return check
    }

    // ------------------------------------------------------------ camera+ARCore
    private fun startCameraAndArCore() {
        try {
            startCameraAndArCoreInner(useArCore = (lens == LensHelper.Lens.MAIN))
        } catch (e: Exception) {
            // Never let a camera/ARCore setup failure kill the app. The
            // user gets a named warning instead of a crash, and the
            // watchdog + crash log carry the details.
            Log.e(TAG, "camera/ARCore setup failed", e)
            onWarning("camera setup failed: ${e.javaClass.simpleName}: ${e.message}")
            // One retry without ARCore: a video + sensors recording is
            // always better than an empty capture.
            if (!fallbackDone) {
                fallbackDone = true
                try {
                    teardownCameraAttempt()
                    startCameraAndArCoreInner(useArCore = false)
                } catch (e2: Exception) {
                    Log.e(TAG, "Camera2 fallback failed", e2)
                    onWarning("camera setup failed: ${e2.javaClass.simpleName}: ${e2.message}")
                }
            }
        }
    }

    /** True once the Camera2-only fallback has been tried (only once). */
    private var fallbackDone = false

    /** Close everything a camera attempt opened, so a retry starts clean. */
    private fun teardownCameraAttempt() {
        try { captureSession?.close() } catch (ignored: Exception) {}
        captureSession = null
        try { cameraDevice?.close() } catch (ignored: Exception) {}
        cameraDevice = null
        try { mediaRecorder?.release() } catch (ignored: Exception) {}
        mediaRecorder = null
        try { arSession?.close() } catch (ignored: Exception) {}
        arSession = null
        sharedCamera = null
        arcoreActive = false
        updateRunning = false
    }

    /**
     * If the camera opened but no video frames arrive within a few seconds,
     * the shared-camera path is silently dead. Tear it down and retry with
     * plain Camera2 (video + sensors). A recording with no route is far
     * better than an empty capture.
     */
    private val cameraWatchdogRunnable = object : Runnable {
        override fun run() {
            if (!running) return
            if (frameCount > 0) return
            if (fallbackDone) return
            if (!arcoreActive) return
            Log.w(TAG, "no frames after 4s with ARCore — falling back to Camera2")
            onWarning("ARCore 未出画面，改用普通相机录制（只有视频，没有路线）")
            fallbackDone = true
            try {
                teardownCameraAttempt()
                startCameraAndArCoreInner(useArCore = false)
            } catch (e: Exception) {
                Log.e(TAG, "Camera2 fallback failed", e)
                onWarning("camera setup failed: ${e.javaClass.simpleName}: ${e.message}")
            }
        }
    }

    private fun startCameraAndArCoreInner(useArCore: Boolean) {
        cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val lenses = LensHelper.enumerateBackCameras(context)
        lensInfo = lenses.firstOrNull { it.lens == lens }
            ?: lenses.firstOrNull()
        val info = lensInfo
        if (info == null) {
            onWarning("no back camera found — recording sensors only")
            return
        }
        // Pick 1920x1080 if the camera offers it, else the largest video size.
        val sizes = info.videoSizes
        videoSize = sizes.firstOrNull { it.width == VIDEO_WIDTH && it.height == VIDEO_HEIGHT }
            ?: sizes.maxByOrNull { it.width * it.height }
            ?: Size(VIDEO_WIDTH, VIDEO_HEIGHT)

        if (useArCore) {
            startArCoreSession()
        } else {
            // Camera2 only. ARCore cannot track on this lens, or the
            // shared-camera path failed and we fell back.
            arcoreActive = false
            trackingState = if (lens == LensHelper.Lens.ULTRAWIDE) {
                "N/A (ultrawide)"
            } else {
                "N/A (Camera2 only)"
            }
            cameraId = info.cameraId
        }

        // MediaRecorder first: its surface must exist before we tell ARCore
        // about our surfaces (setAppSurfaces) and before the capture session.
        try {
            mediaRecorder = createMediaRecorder()
        } catch (e: Exception) {
            Log.e(TAG, "MediaRecorder setup failed", e)
            onWarning("video setup failed: ${e.message}")
            return
        }

        if (arcoreActive) {
            try {
                sharedCamera?.setAppSurfaces(cameraId, listOf(mediaRecorder!!.surface))
            } catch (e: Exception) {
                Log.w(TAG, "setAppSurfaces failed", e)
            }
        }
        openCamera()
        // Watchdog: if no frames flow, fall back to plain Camera2.
        recorderHandler.removeCallbacks(cameraWatchdogRunnable)
        recorderHandler.postDelayed(cameraWatchdogRunnable, 4000)
    }

    /** Create + configure the ARCore session (shared camera). Never throws. */
    private fun startArCoreSession() {
        try {
            val session = Session(context, EnumSet.of(Session.Feature.SHARED_CAMERA))
            val config = Config(session)
            config.focusMode = Config.FocusMode.AUTO
            config.planeFindingMode = Config.PlaneFindingMode.DISABLED
            config.lightEstimationMode = Config.LightEstimationMode.DISABLED
            config.depthMode = Config.DepthMode.DISABLED
            session.configure(config)
            arSession = session
            sharedCamera = session.sharedCamera
            cameraId = session.cameraConfig.cameraId
            arcoreActive = true
            trackingState = "PAUSED"
        } catch (e: UnavailableException) {
            Log.e(TAG, "ARCore session creation failed", e)
            arcoreActive = false
            trackingState = "UNAVAILABLE: ${e.message}"
            onWarning("ARCore unavailable: ${e.message} — recording video + sensors only")
            // Fall back to the main camera without ARCore.
            cameraId = LensHelper.cameraIdFor(context, LensHelper.Lens.MAIN) ?: ""
        } catch (e: Exception) {
            Log.e(TAG, "ARCore session creation failed", e)
            arcoreActive = false
            trackingState = "UNAVAILABLE: ${e.message}"
            onWarning("ARCore unavailable: ${e.message} — recording video + sensors only")
            cameraId = LensHelper.cameraIdFor(context, LensHelper.Lens.MAIN) ?: ""
        }
    }

    private fun createMediaRecorder(): MediaRecorder {
        val mr = if (Build.VERSION.SDK_INT >= 31) MediaRecorder(context)
        else @Suppress("DEPRECATION") MediaRecorder()
        mr.setVideoSource(MediaRecorder.VideoSource.SURFACE)
        mr.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        mr.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
        mr.setVideoSize(videoSize.width, videoSize.height)
        mr.setVideoFrameRate(VIDEO_FPS)
        mr.setVideoEncodingBitRate(VIDEO_BITRATE)
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
            if (arcoreActive) {
                val wrapped = sharedCamera!!.createARDeviceStateCallback(deviceCallback, recorderHandler)
                manager.openCamera(cameraId, wrapped, recorderHandler)
            } else {
                manager.openCamera(cameraId, deviceCallback, recorderHandler)
            }
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
        if (arcoreActive) {
            try {
                surfaces.addAll(sharedCamera!!.arCoreSurfaces)
            } catch (e: Exception) {
                Log.w(TAG, "arCoreSurfaces failed", e)
            }
        }
        surfaces.add(mediaRecorder!!.surface)
        // Live preview (the TextureView on the record screen). Optional:
        // if the preview surface is gone, recording still works.
        val preview = PreviewSurface.surface
        if (preview != null && preview.isValid) {
            surfaces.add(preview)
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
                // Frames are flowing now. Start the recorder, then ARCore.
                try {
                    mediaRecorder?.start()
                } catch (e: Exception) {
                    Log.e(TAG, "MediaRecorder.start failed", e)
                    onWarning("video start failed: ${e.message}")
                }
                // Camera is known now; rewrite meta.json with the real lens.
                writeMeta(final = false)
                if (arcoreActive) {
                    try {
                        arSession?.resume()
                    } catch (e: Exception) {
                        Log.e(TAG, "ARCore resume failed", e)
                        onWarning("ARCore resume failed: ${e.message}")
                        arcoreActive = false
                    }
                    startUpdateLoop()
                }
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {
                Log.e(TAG, "camera capture session configuration failed")
                onWarning("camera capture session configuration failed")
                // The shared-camera path failed to configure. Retry once
                // with plain Camera2 instead of recording nothing.
                if (arcoreActive && !fallbackDone) {
                    fallbackDone = true
                    try {
                        teardownCameraAttempt()
                        startCameraAndArCoreInner(useArCore = false)
                    } catch (e: Exception) {
                        Log.e(TAG, "Camera2 fallback failed", e)
                        onWarning("camera setup failed: ${e.javaClass.simpleName}: ${e.message}")
                    }
                }
            }
        }
        try {
            if (arcoreActive) {
                val wrapped = sharedCamera!!.createARSessionStateCallback(sessionCallback, recorderHandler)
                device.createCaptureSession(surfaces, wrapped, recorderHandler)
            } else {
                device.createCaptureSession(surfaces, sessionCallback, recorderHandler)
            }
        } catch (e: CameraAccessException) {
            Log.e(TAG, "createCaptureSession failed", e)
            onWarning("capture session failed: ${e.message}")
        } catch (e: Exception) {
            Log.e(TAG, "createCaptureSession failed", e)
            onWarning("capture session failed: ${e.javaClass.simpleName}: ${e.message}")
        }
    }

    /** One row per captured video frame. Runs on the recorder thread. */
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

    /** The ARCore update loop. One dedicated thread, one job: copy poses. */
    private fun startUpdateLoop() {
        val session = arSession ?: return
        updateRunning = true
        var lastFrameTimestamp = -1L
        updateThread = Thread({
            while (updateRunning) {
                try {
                    val frame = session.update()
                    val ts = frame.timestamp
                    if (ts != lastFrameTimestamp) {
                        lastFrameTimestamp = ts
                        val camera = frame.camera
                        val pose = camera.pose
                        val tracking = camera.trackingState
                        trackingState = tracking.name
                        val t = (ts - t0Nanos) / 1e9
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
                    }
                } catch (e: Exception) {
                    // update() throws when the session is paused/closed, or on
                    // a transient error. Never let the loop die silently:
                    // log it, and keep looping while we are supposed to run.
                    // If the session was paused by stop(), updateRunning is
                    // false and we exit.
                    if (updateRunning) {
                        Log.w(TAG, "ARCore update error (continuing)", e)
                    }
                }
            }
        }, "arcore-update").apply { start() }
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
        )
    }

    /** Short state line per stream, read by the watchdog when a stream stalls. */
    private fun stateDetail(stream: String): String = when (stream) {
        StreamWatchdog.POSES -> "arcore_active=$arcoreActive tracking=$trackingState update_running=$updateRunning"
        StreamWatchdog.VIDEO_FRAMES -> "camera_open=${cameraDevice != null} session_active=${captureSession != null} frames=$frameCount"
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
            root.put("app_version", "2.0.3")
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
