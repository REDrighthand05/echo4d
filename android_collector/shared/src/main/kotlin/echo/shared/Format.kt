package echo.shared

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.util.Locale

/**
 * Writers for the ECHO capture-folder format, byte-compatible with the
 * readers in echo_scene/sensor_record.py (headers, column order and
 * %.6f float formatting are locked to that module — the Python tests
 * lock the PC recorders to the same contract; the JVM tests here lock
 * this implementation to it, and android_collector/validate_capture.py
 * proves the produced folders against the real Python reader).
 *
 * Format recap (see sensor_record.py for the authoritative text):
 *   meta.json   device, firmware, clock, stream inventory, calibration
 *   frames.csv  frame,t_s,file          file = frame_00001.jpg style
 *   imu.csv     t_s,qw,qx,qy,qz,ax,ay,az,gx,gy,gz   (quat scalar-first,
 *               world-from-device; accel m/s^2 incl. gravity; gyro rad/s)
 *   tof.csv     t_s,z0..z63             8x8 zones, integer mm, -1 = none
 *   poses.csv   t_s,tx,ty,tz,qx,qy,qz,qw,tracking   (sidecar: capture-
 *               device trajectory in the backend's world frame, metres;
 *               quat xyzw; tracking = TRACKING | LIMITED | STOPPED —
 *               how much the pose may be trusted, per row)
 *   frame_meta.csv  frame,t_s,exposure_ns,iso,focus_m,ae_state,af_state
 *               (sidecar: per-frame camera state from the capture
 *               result — the diagnosis stream for blur/exposure)
 *   meta.json "orientation" block (collector v0.10+, optional):
 *               capture_orientation (portrait|landscape),
 *               video_rotation_deg / display_rotation_deg /
 *               sensor_orientation_deg (quarter turns), coded
 *               video_width x video_height — the capture's
 *               orientation stated at record time, validated on the
 *               desktop by sensor_record.parse_orientation.
 *   mag.csv / baro.csv / gps.csv        sidecars, as before
 * All timestamps: seconds on ONE monotonic clock since capture start,
 * strictly increasing per stream (the reader rejects ties/regressions).
 *
 * Durability: rows are STREAMED to disk as they arrive (the v0.1
 * writer buffered everything in memory and wrote at stop(); a killed
 * app lost the whole capture). frames.csv additionally records the
 * byte offset after every row so the stop-time reconciliation can
 * truncate trailing rows that never made it into the encoded video.
 * meta.json is the caller's business: write it at start with
 * finalized=false and rewrite at stop with finalized=true — an
 * unfinalized folder on disk is a recovered capture, not a corrupt
 * one.
 */

const val TOF_GRID = 8
const val TOF_INVALID = -1
const val FORMAT_VERSION = 1

val FRAMES_HEADER = listOf("frame", "t_s", "file")
val IMU_HEADER = listOf("t_s", "qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz")
val TOF_HEADER = listOf("t_s") + (0 until TOF_GRID * TOF_GRID).map { "z$it" }
val POSES_HEADER = listOf("t_s", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "tracking")
val FRAME_META_HEADER = listOf(
    "frame", "t_s", "exposure_ns", "iso", "focus_m", "ae_state", "af_state"
)
val MAG_HEADER = listOf("t_s", "mx", "my", "mz")
val BARO_HEADER = listOf("t_s", "pressure_hpa")
val GPS_HEADER = listOf("t_s", "lat", "lon", "alt_m", "accuracy_m")

fun fmt(x: Double): String = String.format(Locale.US, "%.6f", x)

/**
 * Per-stream stamp discipline, identical in spirit to the PC recorder:
 * stamps are seconds since capture start; a stamp that would tie or
 * regress against the stream's previous stamp is bumped one
 * microsecond past it (the reader demands strictly increasing).
 */
class StampClock(private val originNs: Long) {
    private val last = HashMap<String, Double>()

    /** Convert an absolute monotonic-domain nanosecond timestamp. */
    fun stamp(stream: String, timestampNs: Long): Double {
        var t = (timestampNs - originNs) / 1e9
        val prev = last[stream]
        if (prev != null && t <= prev) t = prev + 1e-6
        last[stream] = t
        return t
    }
}

/** One CSV file being appended row by row. Not thread-safe; the
 *  controller serialises every writer call on its lock. */
private class CsvStream(
    folder: File,
    name: String,
    header: List<String>,
    private val flushEvery: Int,
) {
    private val file = File(folder, name)
    private val out = BufferedOutputStream(FileOutputStream(file))
    private var sinceFlush = 0
    var rows = 0
        private set
    var closed = false
        private set

    init {
        writeRaw(header.joinToString(",") + "\n")
        flush()
    }

    private fun writeRaw(s: String) {
        out.write(s.toByteArray(Charsets.US_ASCII))
    }

    fun row(line: String) {
        writeRaw(line + "\n")
        rows++
        if (++sinceFlush >= flushEvery) flush()
    }

    fun flush() {
        out.flush()
        sinceFlush = 0
    }

    /** Bytes currently in the file (valid right after a flush). */
    fun fileLength(): Long {
        flush()
        return file.length()
    }

    fun truncateTo(bytes: Long) {
        flush()
        out.close()
        RandomAccessFile(file, "rw").use { it.setLength(bytes) }
        closed = true
    }

    fun close() {
        if (!closed) {
            flush()
            out.close()
            closed = true
        }
    }
}

/** Streams one capture folder to disk. Constructing it creates the
 *  three primary CSVs (headers only); sidecars appear on first row.
 *  writeStreams() is the finalize step (idempotent): applies any
 *  pending frame trim, flushes and closes everything. */
class CaptureWriter(private val folder: File) {
    private val frames: CsvStream
    private val imu: CsvStream
    private val tof: CsvStream
    private var poses: CsvStream? = null
    private var frameMeta: CsvStream? = null
    private var mag: CsvStream? = null
    private var baro: CsvStream? = null
    private var gps: CsvStream? = null
    private val frameEndOffsets = ArrayList<Long>()
    private var pendingTrim = -1
    private var finalized = false

    init {
        folder.mkdirs()
        frames = CsvStream(folder, "frames.csv", FRAMES_HEADER, 1)
        imu = CsvStream(folder, "imu.csv", IMU_HEADER, 120)
        tof = CsvStream(folder, "tof.csv", TOF_HEADER, 30)
    }

    val frameCount get() = frames.rows
    val imuCount get() = imu.rows
    val tofCount get() = tof.rows
    val poseCount get() = poses?.rows ?: 0
    val frameMetaCount get() = frameMeta?.rows ?: 0

    fun addFrame(id: Int, t: Double, file: String) {
        frames.row("$id,${fmt(t)},$file")
        frameEndOffsets.add(frames.fileLength())
    }

    fun addImu(t: Double, quatWxyz: DoubleArray, accel: DoubleArray, gyro: DoubleArray) {
        val r = doubleArrayOf(
            t, quatWxyz[0], quatWxyz[1], quatWxyz[2], quatWxyz[3],
            accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]
        )
        imu.row(r.joinToString(",") { fmt(it) })
    }

    fun addTof(t: Double, zonesMm: IntArray) {
        require(zonesMm.size == TOF_GRID * TOF_GRID) { "zones must be 64" }
        tof.row(fmt(t) + "," + zonesMm.joinToString(","))
    }

    /** Pose row: translation (m) + quaternion xyzw in the backend's
     *  world frame, plus the tracker state for this instant
     *  ("TRACKING" / "LIMITED" / "STOPPED"). */
    fun addPose(
        t: Double, tx: Double, ty: Double, tz: Double,
        qx: Double, qy: Double, qz: Double, qw: Double, tracking: String,
    ) {
        val s = poses ?: CsvStream(folder, "poses.csv", POSES_HEADER, 30)
            .also { poses = it }
        val r = doubleArrayOf(t, tx, ty, tz, qx, qy, qz, qw)
        s.row(r.joinToString(",") { fmt(it) } + "," + tracking)
    }

    /** Per-frame camera state (Camera2 capture result values; -1 in
     *  a field means the device did not report it). */
    fun addFrameMeta(
        frame: Int, t: Double, exposureNs: Long, iso: Int,
        focusM: Double, aeState: Int, afState: Int,
    ) {
        val s = frameMeta ?: CsvStream(folder, "frame_meta.csv",
            FRAME_META_HEADER, 30).also { frameMeta = it }
        s.row("$frame,${fmt(t)},$exposureNs,$iso,${fmt(focusM)},$aeState,$afState")
    }

    fun addMag(t: Double, v: DoubleArray) {
        val s = mag ?: CsvStream(folder, "mag.csv", MAG_HEADER, 60)
            .also { mag = it }
        s.row(doubleArrayOf(t, v[0], v[1], v[2]).joinToString(",") { fmt(it) })
    }

    fun addBaro(t: Double, hPa: Double) {
        val s = baro ?: CsvStream(folder, "baro.csv", BARO_HEADER, 60)
            .also { baro = it }
        s.row(doubleArrayOf(t, hPa).joinToString(",") { fmt(it) })
    }

    fun addGps(t: Double, lat: Double, lon: Double, alt: Double, acc: Double) {
        val s = gps ?: CsvStream(folder, "gps.csv", GPS_HEADER, 10)
            .also { gps = it }
        s.row(doubleArrayOf(t, lat, lon, alt, acc).joinToString(",") { fmt(it) })
    }

    /** Drops trailing frame rows beyond [count] at finalize time
     *  (the camera2 backend reconciles harvested SENSOR_TIMESTAMPs
     *  against the number of frames actually encoded into video.mp4;
     *  same for the ARCore backend's codec queue). */
    fun trimFramesTo(count: Int) {
        pendingTrim = count
    }

    /** Pushes buffered rows to the OS (called periodically during a
     *  capture so a process kill loses at most a second of rows). */
    fun flush() {
        frames.flush(); imu.flush(); tof.flush()
        poses?.flush(); frameMeta?.flush()
        mag?.flush(); baro?.flush(); gps?.flush()
    }

    fun writeStreams() {
        if (finalized) return
        finalized = true
        if (pendingTrim >= 0 && pendingTrim < frames.rows) {
            val keepBytes = if (pendingTrim == 0) {
                headerBytes("frames.csv")
            } else {
                frameEndOffsets[pendingTrim - 1]
            }
            frames.truncateTo(keepBytes)
        } else {
            frames.close()
        }
        imu.close(); tof.close()
        poses?.close(); frameMeta?.close()
        mag?.close(); baro?.close(); gps?.close()
    }

    private fun headerBytes(name: String): Long {
        val f = File(folder, name)
        // The header is the first line; its length is stable.
        return f.inputStream().use { ins ->
            var n = 0L
            while (true) {
                val b = ins.read()
                if (b == -1 || b == '\n'.code) { if (b == '\n'.code) n++; break }
                n++
            }
            n
        }
    }

    fun writeMeta(metaJson: String) {
        folder.mkdirs()
        File(folder, "meta.json").writeText(metaJson)
    }
}

/** Minimal JSON writer (objects keep insertion order; enough for
 *  meta.json, which the Python side only parses). */
class JsonObj {
    private val entries = ArrayList<Pair<String, Any?>>()
    fun put(key: String, value: Any?): JsonObj { entries.add(key to value); return this }
    fun render(indent: Int = 0): String {
        val pad = "  ".repeat(indent)
        val padIn = "  ".repeat(indent + 1)
        val sb = StringBuilder("{\n")
        entries.forEachIndexed { i, (k, v) ->
            sb.append(padIn).append('"').append(escape(k)).append("\": ")
                .append(renderValue(v, indent + 1))
            if (i < entries.size - 1) sb.append(',')
            sb.append('\n')
        }
        sb.append(pad).append('}')
        return sb.toString()
    }

    companion object {
        fun escape(s: String): String = s.replace("\\", "\\\\").replace("\"", "\\\"")
            .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\t")

        fun renderValue(v: Any?, indent: Int): String = when (v) {
            null -> "null"
            is JsonObj -> v.render(indent)
            is String -> "\"${escape(v)}\""
            is Boolean -> v.toString()
            is Int, is Long -> v.toString()
            is Float -> renderValue(v.toDouble(), indent)
            is Double -> if (v == v.toLong().toDouble() && Math.abs(v) < 1e15) {
                v.toLong().toString()
            } else v.toString()
            is List<*> -> v.joinToString(", ", "[", "]") { renderValue(it, indent) }
            else -> "\"${escape(v.toString())}\""
        }
    }
}
