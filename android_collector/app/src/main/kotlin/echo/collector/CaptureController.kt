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
 * Backend selection (capability-adaptive depth):
 *   1. chosen camera has DEPTH16  -> Camera2 backend, depth from it
 *   2. else ARCore Depth works   -> ARCore backend (video + poses +
 *                                   depth from the ARCore session)
 *   3. else                      -> Camera2 backend, tof.csv empty
 * The IMU/extras streams (ImuStreams) run alongside either backend.
 * Every writer mutation is serialised on one lock — the backends and
 * the sensor listener run on different threads.
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
            caps.arcoreDepthSupported -> "arcore"
            else -> "camera2"
        }

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
    private var harvestedFrames: Int = 0

    val frameCount get() = synchronized(lock) { writer?.frameCount ?: 0 }
    val imuCount get() = synchronized(lock) { writer?.imuCount ?: 0 }
    val tofCount get() = synchronized(lock) { writer?.tofCount ?: 0 }
    val elapsedMs get() =
        if (state == State.RECORDING) SystemClock.elapsedRealtime() - startElapsedMs else 0L

    fun storageLeftBytes(): Long {
        val base = context.getExternalFilesDir(null) ?: context.filesDir
        return StatFs(base.path).availableBytes
    }

    fun start() {
        check(state == State.IDLE)
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(
            File(context.getExternalFilesDir(null), "captures"),
            "capture_$stamp"
        )
        dir.mkdirs()
        folder = dir
        val w = CaptureWriter(dir)
        writer = w
        val clk = StampClock(SystemClock.elapsedRealtimeNanos())
        clock = clk
        frameSeq = 0
        notes.clear()
        encodedFrames = -1
        harvestedFrames = 0

        val sink = object : ImuStreams.Sink {
            override fun onImuRow(tNs: Long, q: DoubleArray, a: DoubleArray, g: DoubleArray) {
                synchronized(lock) { w.addImu(clk.stamp("imu", tNs), q, a, g) }
            }
            override fun onMag(tNs: Long, v: DoubleArray) {
                synchronized(lock) {
                    w.magRows.add(doubleArrayOf(clk.stamp("mag", tNs), v[0], v[1], v[2]))
                }
            }
            override fun onBaro(tNs: Long, hPa: Double) {
                synchronized(lock) { w.baroRows.add(doubleArrayOf(clk.stamp("baro", tNs), hPa)) }
            }
            override fun onGps(tNs: Long, lat: Double, lon: Double, alt: Double, acc: Double) {
                synchronized(lock) {
                    w.gpsRows.add(doubleArrayOf(clk.stamp("gps", tNs), lat, lon, alt, acc))
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
                harvestedFrames = frameSeq
                w.addFrame(frameSeq, clk.stamp("frames", tsNs),
                    "frame_%05d.jpg".format(frameSeq))
            }
        }
        val onDepth: (Long, IntArray) -> Unit = { tsNs, zones ->
            synchronized(lock) { w.addTof(clk.stamp("tof", tsNs), zones) }
        }

        when (backendName) {
            "arcore" -> {
                val rec = ArCoreRecorder(context, dir, onFrame,
                    { tsNs, pose: Pose ->
                        synchronized(lock) {
                            val tr = pose.translation
                            val rq = pose.rotationQuaternion
                            w.posesRows.add(doubleArrayOf(
                                clk.stamp("poses", tsNs),
                                tr[0].toDouble(), tr[1].toDouble(), tr[2].toDouble(),
                                rq[0].toDouble(), rq[1].toDouble(), rq[2].toDouble(),
                                rq[3].toDouble()
                            ))
                        }
                    },
                    onDepth)
                arcore = rec
                rec.start(onError)
            }
            else -> {
                val rec = Camera2Recorder(
                    context, caps.chosenCameraId, dir,
                    withDepth = backendName == "camera2+depth16",
                    onFrameStamp = onFrame, onDepth = onDepth
                )
                camera2 = rec
                rec.start(onError)
            }
        }
        startElapsedMs = SystemClock.elapsedRealtime()
        state = State.RECORDING
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
            it.depthSize?.let { d ->
                notes.add("depth16 ${d.width}x${d.height}")
            }
        }
        arcore?.let {
            it.stop()
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
            w.writeMeta(buildMeta(w).render())
        }
        state = State.IDLE
        return folder!!
    }

    private fun buildMeta(w: CaptureWriter): JsonObj {
        val duration = elapsedMs / 1000.0
        val camFd = Convert.camFromDevice(caps.chosenSensorOrientation)
        val tofFov = camera2?.depthFovDeg ?: 45.0
        fun hz(count: Int) = if (duration > 0) count / duration else 0.0
        val meta = JsonObj()
            .put("format_version", echo.shared.FORMAT_VERSION)
            .put("device", JsonObj()
                .put("name", "ECHO Collector")
                .put("board", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
                .put("firmware", "android-collector 0.1"))
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
                    .put("present", w.tofCount > 0)))
            .put("calibration", JsonObj()
                .put("cam_from_device", camFd.map { it.toList() })
                .put("camera", null))
            .put("android", JsonObj()
                .put("backend", backendName)
                .put("sensor_orientation_deg", caps.chosenSensorOrientation)
                .put("capabilities", caps.manifest)
                .put("notes", notes.toList()))
        return meta
    }
}
