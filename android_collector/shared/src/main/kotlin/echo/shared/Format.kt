package echo.shared

import java.io.File
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
 * All timestamps: seconds on ONE monotonic clock since capture start,
 * strictly increasing per stream (the reader rejects ties/regressions).
 */

const val TOF_GRID = 8
const val TOF_INVALID = -1
const val FORMAT_VERSION = 1

val FRAMES_HEADER = listOf("frame", "t_s", "file")
val IMU_HEADER = listOf("t_s", "qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz")
val TOF_HEADER = listOf("t_s") + (0 until TOF_GRID * TOF_GRID).map { "z$it" }

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

/** Accumulates streams and writes one capture folder on finish(). */
class CaptureWriter(private val folder: File) {
    private val frameIds = ArrayList<Int>()
    private val frameT = ArrayList<Double>()
    private val frameFiles = ArrayList<String>()
    private val imuRows = ArrayList<DoubleArray>()   // t + 10 values
    private val tofRows = ArrayList<Pair<Double, IntArray>>()
    val posesRows = ArrayList<DoubleArray>()          // t,tx,ty,tz,qx,qy,qz,qw (sidecar)
    val magRows = ArrayList<DoubleArray>()            // t,mx,my,mz (sidecar)
    val baroRows = ArrayList<DoubleArray>()           // t,hPa (sidecar)
    val gpsRows = ArrayList<DoubleArray>()            // t,lat,lon,alt,acc (sidecar)

    fun addFrame(id: Int, t: Double, file: String) {
        frameIds.add(id); frameT.add(t); frameFiles.add(file)
    }

    fun addImu(t: Double, quatWxyz: DoubleArray, accel: DoubleArray, gyro: DoubleArray) {
        imuRows.add(
            doubleArrayOf(
                t, quatWxyz[0], quatWxyz[1], quatWxyz[2], quatWxyz[3],
                accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]
            )
        )
    }

    fun addTof(t: Double, zonesMm: IntArray) {
        require(zonesMm.size == TOF_GRID * TOF_GRID) { "zones must be 64" }
        tofRows.add(t to zonesMm.copyOf())
    }

    val frameCount get() = frameIds.size
    val imuCount get() = imuRows.size
    val tofCount get() = tofRows.size

    /** Drops trailing frame rows beyond [count] (the camera2 backend
     *  reconciles harvested SENSOR_TIMESTAMPs against the number of
     *  frames actually encoded into video.mp4). */
    fun trimFramesTo(count: Int) {
        while (frameIds.size > count) {
            frameIds.removeAt(frameIds.size - 1)
            frameT.removeAt(frameT.size - 1)
            frameFiles.removeAt(frameFiles.size - 1)
        }
    }

    /** Writes frames.csv / imu.csv / tof.csv (+ sidecars that are
     *  non-empty). meta.json is written by writeMeta with the caller's
     *  meta text (built by Json.kt) so meta policy stays with the app /
     *  harness, byte layout with the format. */
    fun writeStreams() {
        folder.mkdirs()
        File(folder, "frames.csv").bufferedWriter().use { w ->
            w.write(FRAMES_HEADER.joinToString(",")); w.newLine()
            for (i in frameIds.indices) {
                w.write("${frameIds[i]},${fmt(frameT[i])},${frameFiles[i]}"); w.newLine()
            }
        }
        File(folder, "imu.csv").bufferedWriter().use { w ->
            w.write(IMU_HEADER.joinToString(",")); w.newLine()
            for (r in imuRows) {
                w.write(r.joinToString(",") { fmt(it) }); w.newLine()
            }
        }
        File(folder, "tof.csv").bufferedWriter().use { w ->
            w.write(TOF_HEADER.joinToString(",")); w.newLine()
            for ((t, z) in tofRows) {
                w.write(fmt(t)); w.write(",")
                w.write(z.joinToString(",")); w.newLine()
            }
        }
        if (posesRows.isNotEmpty()) {
            File(folder, "poses.csv").bufferedWriter().use { w ->
                w.write("t_s,tx,ty,tz,qx,qy,qz,qw"); w.newLine()
                for (r in posesRows) { w.write(r.joinToString(",") { fmt(it) }); w.newLine() }
            }
        }
        if (magRows.isNotEmpty()) {
            File(folder, "mag.csv").bufferedWriter().use { w ->
                w.write("t_s,mx_ut,my_ut,mz_ut"); w.newLine()
                for (r in magRows) { w.write(r.joinToString(",") { fmt(it) }); w.newLine() }
            }
        }
        if (baroRows.isNotEmpty()) {
            File(folder, "baro.csv").bufferedWriter().use { w ->
                w.write("t_s,pressure_hpa"); w.newLine()
                for (r in baroRows) { w.write(r.joinToString(",") { fmt(it) }); w.newLine() }
            }
        }
        if (gpsRows.isNotEmpty()) {
            File(folder, "gps.csv").bufferedWriter().use { w ->
                w.write("t_s,lat,lon,alt_m,accuracy_m"); w.newLine()
                for (r in gpsRows) { w.write(r.joinToString(",") { fmt(it) }); w.newLine() }
            }
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
            .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

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
