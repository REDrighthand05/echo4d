package echo.collector

import android.content.Context
import android.os.Handler
import android.os.HandlerThread
import android.os.StatFs
import android.os.SystemClock
import com.google.ar.core.Pose
import echo.shared.CaptureWriter
import echo.shared.Convert
import echo.shared.JsonObj
import echo.shared.StampClock
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Owns one capture: the folder, the shared StampClock (origin =
 * SystemClock.elapsedRealtimeNanos at start — the same nanosecond
 * domain Camera2 SENSOR_TIMESTAMP and SensorEvent timestamps live
 * in), the format writer, and the backends.
 *
 * Backend selection (capability-adaptive, pose-first since v0.2):
 *   1. chosen camera has DEPTH16  -> Camera2 backend, measured depth
 *   2. else ARCore is supported   -> ARCore backend (video + 6DoF
 *      poses, depth too when the device offers it). Poses are a
 *      primary stream, not a bonus: a metric trajectory is worth
 *      more to the reconstruction than a second video path, and
 *      ARCore cannot share the camera with Camera2, so the pose
 *      backend must own it whenever it exists.
 *   3. else                       -> Camera2 backend, tof.csv empty
 * The IMU/extras streams (ImuStreams) run alongside either backend.
 * Every writer mutation is serialised on one lock — the backends and
 * the sensor listener run on different threads.
 *
 * Durability: meta.json is written at start (finalized=false) and
 * rewritten at stop (finalized=true); the writer streams rows to
 * disk as they arrive, so a killed app leaves a usable, honestly
 * marked capture instead of nothing.
 */
class CaptureController(
    private val context: Context,
    private val caps: Capabilities,
    private val onError: (String) -> Unit,
) {
    enum class State { IDLE, RECORDING }

    @Volatile var state = State.IDLE
        private set
    var folder: File? = null
        private set
    val backendName: String
        get() = when {
            caps.camera2Depth16 -> "camera2+depth16"
            caps.arcoreSupported -> "arcore"
            else -> "camera2"
        }

    /** The backend that actually ran (set during start()). The
     *  planned backend can lose to reality — ARCore bring-up
     *  failing falls back to Camera2 — and meta must say what
     *  really happened, not what was hoped. */
    @Volatile var activeBackend: String = "none"
        private set

    /** Fired (on the caller's thread) when storage runs out mid-
     *  capture and the controller has stopped itself. */
    var onAutoStop: ((File) -> Unit)? = null

    /** Preview stills from whichever backend owns the camera
     *  (~5 Hz, small bitmaps). The service wires this to its
     *  shared latest-preview slot for the Activity to show. */
    var onPreviewFrame: ((android.graphics.Bitmap) -> Unit)? = null

    @Volatile private var latestQuat: DoubleArray? = null
    private val trailPoints = ArrayDeque<DoubleArray>()
    private var frameSink: ((Long) -> Unit)? = null
    private var depthSink: ((Long, IntArray) -> Unit)? = null
    private var backendSinceMs = 0L
    private var watchdogRetried = false

    /** Latest rotation-vector quaternion (wxyz) for the UI's
     *  attitude indicator; null before the first IMU row. */
    fun orientation(): DoubleArray? = latestQuat?.copyOf()

    /** Top-down trajectory so far (ARCore x/z pairs), oldest
     *  first, capped — the UI's route trace. */
    fun trail(): List<DoubleArray> =
        synchronized(lock) { trailPoints.toList() }

    private val lock = Any()
    private var writer: CaptureWriter? = null
    private var clock: StampClock? = null
    private var frameSeq = 0
    private var startElapsedMs = 0L
    private var camera2: Camera2Recorder? = null
    private var arcore: ArCoreRecorder? = null
    private var imu: ImuStreams? = null
    private var imuThread: HandlerThread? = null
    private val notes = ArrayList<String>()
    private var encodedFrames: Int = -1

    val frameCount get() = synchronized(lock) { writer?.frameCount ?: 0 }
    val imuCount get() = synchronized(lock) { writer?.imuCount ?: 0 }
    val tofCount get() = synchronized(lock) { writer?.tofCount ?: 0 }
    val poseCount get() = synchronized(lock) { writer?.poseCount ?: 0 }
    val elapsedMs get() =
        if (state == State.RECORDING) SystemClock.elapsedRealtime() - startElapsedMs else 0L

    fun storageLeftBytes(): Long {
        val base = Storage.base(context)
        return StatFs(base.path).availableBytes
    }

    fun start() {
        check(state == State.IDLE)
        val free = storageLeftBytes()
        require(free > MIN_FREE_START_BYTES) {
            "存储空间不足（剩 %.1f GB，至少需要 %.1f GB）".format(
                free / 1e9, MIN_FREE_START_BYTES / 1e9)
        }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(Storage.capturesDir(context), "capture_$stamp")
        dir.mkdirs()
        folder = dir
        val w = CaptureWriter(dir)
        writer = w
        val clk = StampClock(SystemClock.elapsedRealtimeNanos())
        clock = clk
        frameSeq = 0
        notes.clear()
        encodedFrames = -1
        latestQuat = null
        synchronized(lock) { trailPoints.clear() }
        // The capture exists on disk from this moment on.
        w.writeMeta(buildMeta(w, finalized = false).render())

        val sink = object : ImuStreams.Sink {
            override fun onImuRow(tNs: Long, q: DoubleArray, a: DoubleArray, g: DoubleArray) {
                latestQuat = q.copyOf()
                synchronized(lock) { w.addImu(clk.stamp("imu", tNs), q, a, g) }
            }
            override fun onMag(tNs: Long, v: DoubleArray) {
                synchronized(lock) { w.addMag(clk.stamp("mag", tNs), v) }
            }
            override fun onBaro(tNs: Long, hPa: Double) {
                synchronized(lock) { w.addBaro(clk.stamp("baro", tNs), hPa) }
            }
            override fun onGps(tNs: Long, lat: Double, lon: Double, alt: Double, acc: Double) {
                synchronized(lock) {
                    w.addGps(clk.stamp("gps", tNs), lat, lon, alt, acc)
                }
            }
        }
        imuThread = HandlerThread("echo-imu").apply { start() }
        val streams = ImuStreams(context, sink)
        imu = streams
        streams.start(Handler(imuThread!!.looper))

        val onFrame: (Long) -> Unit = { tsNs ->
            synchronized(lock) {
                frameSeq++
                w.addFrame(frameSeq, clk.stamp("frames", tsNs),
                    "frame_%05d.jpg".format(frameSeq))
            }
        }
        val onDepth: (Long, IntArray) -> Unit = { tsNs, zones ->
            synchronized(lock) { w.addTof(clk.stamp("tof", tsNs), zones) }
        }
        frameSink = onFrame
        depthSink = onDepth
        watchdogRetried = false

        // Decide the backend BEFORE anything records. ARCore is
        // brought up synchronously; if it cannot start on this
        // device, fall back to Camera2 right here — the capture
        // then has video+IMU but no poses, instead of the v0.2
        // failure mode (sensors happily recording, camera never
        // opened, user walks a route for nothing).
        var arcoreReady = false
        if (backendName == "arcore") {
            val rec = ArCoreRecorder(context, dir, onFrame,
                { tsNs, pose: Pose, tracking: String ->
                    synchronized(lock) {
                        val tr = pose.translation
                        val rq = pose.rotationQuaternion
                        w.addPose(
                            clk.stamp("poses", tsNs),
                            tr[0].toDouble(), tr[1].toDouble(), tr[2].toDouble(),
                            rq[0].toDouble(), rq[1].toDouble(), rq[2].toDouble(),
                            rq[3].toDouble(), tracking
                        )
                        trailPoints.addLast(doubleArrayOf(
                            tr[0].toDouble(), tr[2].toDouble()))
                        while (trailPoints.size > 1200) {
                            trailPoints.removeFirst()
                        }
                    }
                },
                onDepth,
                { bmp -> onPreviewFrame?.invoke(bmp) })
            val err = rec.prepareBlocking(onError)
            if (err == null) {
                arcore = rec
                arcoreReady = true
                activeBackend = "arcore"
                notes.add(rec.note)
            } else {
                notes.add(
                    "arcore unavailable: $err; fell back to camera2 " +
                        "(video+imu, no poses this capture)")
                activeBackend = "camera2"
            }
        } else {
            activeBackend = backendName
        }

        backendSinceMs = SystemClock.elapsedRealtime()
        when {
            arcoreReady -> {
                // prepareBlocking already started the loop thread.
            }
            else -> {
                startCamera2Backend()
            }
        }
        startElapsedMs = SystemClock.elapsedRealtime()
        state = State.RECORDING
    }

    /** Starts (or restarts, from the watchdog) the Camera2
     *  backend against the live writer/clock/folder. */
    private fun startCamera2Backend() {
        val w = writer!!
        val clk = clock!!
        val rec = Camera2Recorder(
            context, caps, folder!!,
            withDepth = backendName == "camera2+depth16",
            onFrameStamp = frameSink!!, onDepth = depthSink!!,
            onFrameMeta = { seq, tsNs, exposureNs, iso, focusM, ae, af ->
                synchronized(lock) {
                    w.addFrameMeta(
                        seq, clk.stamp("framemeta", tsNs),
                        exposureNs, iso, focusM, ae, af)
                }
            },
            onPreview = { bmp -> onPreviewFrame?.invoke(bmp) }
        )
        camera2 = rec
        rec.start(onError)
    }

    /** Periodic housekeeping while recording (the UI/service ticker
     *  calls this about once a second): flush rows to disk and stop
     *  cleanly before the disk is completely full. */
    fun tick() {
        if (state != State.RECORDING) return
        synchronized(lock) { writer?.flush() }
        if (storageLeftBytes() < MIN_FREE_STOP_BYTES) {
            notes.add("auto-stopped: storage below %.0f MB floor".format(
                MIN_FREE_STOP_BYTES / 1e6))
            val f = stop()
            onAutoStop?.invoke(f)
            return
        }
        // Camera watchdog: a backend that never produces a frame
        // must not let the user walk a whole route for nothing.
        // First offence on ARCore: heal in place — same folder,
        // same clock, IMU keeps running, only the camera backend
        // is swapped for Camera2. Only if the swap also produces
        // nothing does the capture stop, loudly.
        if (frameCount == 0 &&
            SystemClock.elapsedRealtime() - backendSinceMs >
                NO_FRAME_GRACE_MS) {
            if (activeBackend == "arcore" && !watchdogRetried) {
                watchdogRetried = true
                notes.add(
                    "watchdog: arcore produced no frames; swapped " +
                        "to camera2 mid-capture")
                try { arcore?.stop() } catch (e: Exception) { /* swap on */ }
                arcore?.let { notes.add(it.note) }
                arcore = null
                activeBackend = "camera2"
                startCamera2Backend()
                backendSinceMs = SystemClock.elapsedRealtime()
            } else {
                notes.add(
                    "watchdog: no frames after ${elapsedMs / 1000}s; " +
                        "stopped — the camera produced nothing")
                val f = stop()
                onError("相机没有出画面，已自动停止——这次没有录上")
                onAutoStop?.invoke(f)
            }
        }
    }

    fun stop(): File {
        check(state == State.RECORDING)
        imu?.stop()
        imuThread?.quitSafely()
        var camNote = ""
        camera2?.let {
            val (count, note) = it.stop()
            encodedFrames = count
            camNote = note
            notes.add("focus: ${it.focusNote}")
            notes.add("video ${it.videoSize?.width}x${it.videoSize?.height}; $note")
            notes.add("lens: ${it.lensNote}")
            it.depthSize?.let { d ->
                notes.add("depth16 ${d.width}x${d.height}")
            }
        }
        arcore?.let {
            encodedFrames = it.stop()
            notes.add(it.note)
        }
        val w = writer!!
        synchronized(lock) {
            if (encodedFrames >= 0 && encodedFrames < w.frameCount) {
                notes.add(
                    "frames.csv trimmed ${w.frameCount} -> $encodedFrames " +
                        "to match encoded video frames")
                w.trimFramesTo(encodedFrames)
            }
            w.writeStreams()
            w.writeMeta(buildMeta(w, finalized = true).render())
        }
        state = State.IDLE
        return folder!!
    }

    private fun buildMeta(w: CaptureWriter, finalized: Boolean): JsonObj {
        val duration = elapsedMs / 1000.0
        val camFd = Convert.camFromDevice(caps.chosenSensorOrientation)
        val tofFov = camera2?.depthFovDeg ?: 45.0
        fun hz(count: Int) = if (duration > 0) count / duration else 0.0
        val intrinsics = camera2?.intrinsicsJson() ?: arcore?.intrinsicsJson()
        val meta = JsonObj()
            .put("format_version", echo.shared.FORMAT_VERSION)
            .put("device", JsonObj()
                .put("name", "ECHO Collector")
                .put("board", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
                .put("firmware", "android-collector 0.6"))
            .put("clock", JsonObj()
                .put("type", "monotonic")
                .put("unit", "s")
                .put("origin_utc",
                    java.time.Instant.now().toString())
                .put("origin_note", "t=0 is capture start on the Android " +
                    "elapsedRealtime clock; camera frames are stamped by " +
                    "their SENSOR_TIMESTAMP and sensor events by their " +
                    "own timestamps — one shared nanosecond domain, no " +
                    "arrival stamping anywhere"))
            .put("duration_s", Math.round(duration * 1000.0) / 1000.0)
            .put("finalized", finalized)
            .put("streams", JsonObj()
                .put("video", JsonObj()
                    .put("file", "video.mp4")
                    .put("fps_nominal", 30.0)
                    .put("frames_csv", "frames.csv"))
                .put("imu", JsonObj()
                    .put("hz", hz(w.imuCount))
                    .put("quat_order", "wxyz")
                    .put("accel_unit", "m/s^2 (device frame, incl. gravity)")
                    .put("gyro_unit", "rad/s (device frame)")
                    .put("note", "one row per rotation-vector event; " +
                        "accel/gyro are the latest samples at that " +
                        "instant (latest-sample hold)"))
                .put("tof", JsonObj()
                    .put("hz", hz(w.tofCount))
                    .put("grid", listOf(8, 8))
                    .put("fov_deg", tofFov)
                    .put("max_range_mm", 8191)
                    .put("invalid", -1)
                    .put("unit", "mm")
                    .put("present", w.tofCount > 0))
                .put("poses", JsonObj()
                    .put("file", "poses.csv")
                    .put("present", w.poseCount > 0)
                    .put("hz", hz(w.poseCount))
                    .put("frame", "backend world frame (ARCore: metres, " +
                        "+Y up, right-handed)")
                    .put("quat_order", "xyzw")
                    .put("note", "per-frame tracker pose with its " +
                        "tracking state; rows stamped on the shared " +
                        "clock, aligned with frames.csv by time"))
                .put("frame_meta", JsonObj()
                    .put("file", "frame_meta.csv")
                    .put("present", w.frameMetaCount > 0)
                    .put("note", "per-frame exposure/ISO/focus/AE/AF " +
                        "from the capture result (Camera2 backend)")))
            .put("calibration", JsonObj()
                .put("cam_from_device", camFd.map { it.toList() })
                .put("camera", intrinsics))
            .put("android", JsonObj()
                .put("backend",
                    if (activeBackend != "none") activeBackend
                    else backendName)
                .put("backend_planned", backendName)
                .put("sensor_orientation_deg", caps.chosenSensorOrientation)
                .put("capabilities", caps.manifest)
                .put("notes", notes.toList()))
        return meta
    }

    companion object {
        const val MIN_FREE_START_BYTES = 1_500_000_000L
        const val MIN_FREE_STOP_BYTES = 300_000_000L
        const val NO_FRAME_GRACE_MS = 3_000L
    }
}
