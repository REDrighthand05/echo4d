package com.echo.collector

import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Stream health watchdog + post-record self-check.
 *
 * This is the answer to "never again 'I don't know why the data stopped'".
 *
 * During recording:
 * - Every stream (video frames, ARCore poses, IMU, magnetometer, light,
 *   barometer) reports each sample via [noteSample]. The watchdog keeps
 *   the last-sample time, a count, first/last timestamps and the largest
 *   gap between consecutive samples for every stream.
 * - Once a second it checks: if a stream that should be producing data
 *   has produced nothing for more than [STALE_THRESHOLD_MS], it fires
 *   [onStale] exactly once per stale episode (naming the stream, how long
 *   it has been silent, and the current state detail the session
 *   supplies), writes the event to watchdog.log, and logs it. As soon as
 *   the stream produces a sample again the flag clears, so a recovery is
 *   visible too.
 * - A stream is only watched after its first sample arrives (so the
 *   light sensor, which is slow, does not false-positive at start-up,
 *   and the barometer is only watched if the sensor exists).
 *
 * After recording, [snapshot] produces the self-check: per-stream counts,
 * first/last timestamps, coverage against the video duration, largest gap,
 * and a status (ok / warning / missing). Anything below 90% pose coverage
 * or with a gap over 1 s is flagged.
 */
class StreamWatchdog {

    companion object {
        const val VIDEO_FRAMES = "video_frames"
        const val POSES = "poses"
        const val IMU = "imu"
        const val MAGNETOMETER = "magnetometer"
        const val LIGHT = "light"
        const val BAROMETER = "barometer"

        const val STALE_THRESHOLD_MS = 3000L
        private const val CHECK_INTERVAL_MS = 1000L
        private const val TAG = "StreamWatchdog"
    }

    /** Called once per stale episode: (stream, silentMs, stateDetail). */
    var onStale: ((stream: String, silentMs: Long, detail: String) -> Unit)? = null

    /** Called when a stalled stream produces a sample again. */
    var onRecovered: ((stream: String) -> Unit)? = null

    /** The session supplies a short state line per stream (tracking state,
     *  camera state, listener state). Read when a stream goes stale. */
    var stateDetailProvider: ((stream: String) -> String)? = null

    private data class Stats(
        var count: Long = 0,
        var firstNanos: Long = -1,
        var lastNanos: Long = -1,
        var prevNanos: Long = -1,
        var maxGapNanos: Long = 0,
    )

    private val stats = ConcurrentHashMap<String, Stats>()
    private val staleFlagged = ConcurrentHashMap<String, Boolean>()
    private var logFile: File? = null
    private var t0Nanos: Long = -1

    private var checkThread: HandlerThread? = null
    private var checkHandler: Handler? = null
    private val checkRunnable = object : Runnable {
        override fun run() {
            checkOnce()
            checkHandler?.postDelayed(this, CHECK_INTERVAL_MS)
        }
    }

    /** t0 is the recording start (elapsedRealtimeNanos). Used to turn
     *  absolute timestamps into t_s for the report. */
    fun setT0(t0: Long) {
        t0Nanos = t0
    }

    fun setLogFile(file: File) {
        logFile = file
    }

    /** Register a stream as expected (watched once its first sample lands). */
    fun expect(stream: String) {
        stats.putIfAbsent(stream, Stats())
    }

    /** Record one sample. Cheap: a few long comparisons. Safe from any thread. */
    fun noteSample(stream: String, timestampNanos: Long) {
        val s = stats.computeIfAbsent(stream) { Stats() }
        synchronized(s) {
            if (s.count == 0L) s.firstNanos = timestampNanos
            if (s.prevNanos > 0) {
                val gap = timestampNanos - s.prevNanos
                if (gap > s.maxGapNanos) s.maxGapNanos = gap
            }
            s.prevNanos = timestampNanos
            s.lastNanos = timestampNanos
            s.count++
        }
        // Stream recovered: clear the stale flag so a later stall re-alerts.
        if (staleFlagged[stream] == true) {
            staleFlagged[stream] = false
            log("RECOVERED stream=$stream")
            onRecovered?.invoke(stream)
        }
    }

    fun start() {
        val t = HandlerThread("watchdog").also { it.start() }
        checkThread = t
        checkHandler = Handler(t.looper)
        checkHandler?.postDelayed(checkRunnable, CHECK_INTERVAL_MS)
    }

    fun stop() {
        checkHandler?.removeCallbacks(checkRunnable)
        checkThread?.quitSafely()
        checkHandler = null
        checkThread = null
    }

    private fun checkOnce() {
        val now = SystemClock.elapsedRealtimeNanos()
        for ((stream, s) in stats) {
            val last: Long
            val count: Long
            synchronized(s) {
                last = s.lastNanos
                count = s.count
            }
            if (count == 0L) continue // not started yet; nothing to judge
            val silentMs = (now - last) / 1_000_000L
            if (silentMs > STALE_THRESHOLD_MS && staleFlagged[stream] != true) {
                staleFlagged[stream] = true
                val detail = stateDetailProvider?.invoke(stream) ?: ""
                val msg = "STALE stream=$stream silent=${silentMs}ms $detail"
                Log.w(TAG, msg)
                log(msg)
                onStale?.invoke(stream, silentMs, detail)
            }
        }
    }

    private fun log(line: String) {
        val f = logFile ?: return
        try {
            f.appendText("${System.currentTimeMillis()} $line\n")
        } catch (ignored: Exception) {
        }
    }

    /** Per-stream result for the post-record self-check. */
    data class StreamResult(
        val count: Long,
        val firstTs: Double,
        val lastTs: Double,
        val coveragePct: Double,
        val maxGapS: Double,
        val status: String,
    )

    data class SelfCheck(
        val videoDurationS: Double,
        val streams: Map<String, StreamResult>,
        val overall: String,
        val warnings: List<String>,
    ) {
        fun toJson(): JSONObject {
            val root = JSONObject()
            root.put("video_duration_s", videoDurationS)
            val obj = JSONObject()
            for ((name, r) in streams) {
                val o = JSONObject()
                o.put("count", r.count)
                o.put("first_t", r.firstTs)
                o.put("last_t", r.lastTs)
                o.put("coverage_pct", r.coveragePct)
                o.put("max_gap_s", r.maxGapS)
                o.put("status", r.status)
                obj.put(name, o)
            }
            root.put("streams", obj)
            root.put("overall", overall)
            root.put("warnings", org.json.JSONArray(warnings))
            return root
        }
    }

    /**
     * Build the post-record self-check. [videoDurationS] is the video
     * length in seconds (last frame t_s). Coverage is the stream's time
     * span divided by the video duration.
     */
    fun snapshot(videoDurationS: Double): SelfCheck {
        val results = LinkedHashMap<String, StreamResult>()
        val warnings = mutableListOf<String>()
        for ((stream, s) in stats) {
            val count: Long
            val first: Long
            val last: Long
            val maxGap: Long
            synchronized(s) {
                count = s.count
                first = s.firstNanos
                last = s.lastNanos
                maxGap = s.maxGapNanos
            }
            val t0 = if (t0Nanos > 0) t0Nanos else first
            val firstTs = if (count > 0) (first - t0) / 1e9 else 0.0
            val lastTs = if (count > 0) (last - t0) / 1e9 else 0.0
            val span = lastTs - firstTs
            val coverage = if (videoDurationS > 0) (span / videoDurationS * 100.0) else 0.0
            val status: String
            when {
                count == 0L -> {
                    status = "missing"
                    warnings.add("$stream: no samples at all")
                }
                stream == POSES && coverage < 90.0 -> {
                    status = "warning"
                    warnings.add("poses cover only ${"%.1f".format(coverage)}% of the video")
                }
                maxGap > 1_000_000_000L -> {
                    status = "warning"
                    warnings.add("$stream: largest gap ${"%.2f".format(maxGap / 1e9)}s")
                }
                else -> status = "ok"
            }
            results[stream] = StreamResult(
                count = count,
                firstTs = firstTs,
                lastTs = lastTs,
                coveragePct = coverage,
                maxGapS = maxGap / 1e9,
                status = status,
            )
        }
        val overall = if (warnings.isEmpty()) "ok" else "warning"
        return SelfCheck(videoDurationS, results, overall, warnings)
    }
}
