package com.echo.collector

import android.content.Context
import android.os.Environment
import org.json.JSONObject
import java.io.File

/**
 * Recording management: list past captures, compute their stats, delete.
 *
 * A capture is a folder under EchoCaptures/ containing video.mp4,
 * frames.csv, poses.csv and meta.json. Stats are computed from the files
 * themselves (no separate database to go stale):
 * - video duration: last t_s in frames.csv (falls back to meta.json).
 * - pose count: data rows in poses.csv (header excluded).
 * - pose coverage %: pose time span / video duration.
 * - size: the folder's total bytes.
 *
 * All work runs on a background thread; callers get the list via callback.
 */
object CaptureStore {

    data class CaptureInfo(
        val name: String,
        val dir: File,
        val dateMs: Long,
        val sizeBytes: Long,
        val videoDurationS: Double,
        val poseCount: Long,
        val poseCoveragePct: Double,
    )

    fun capturesRoot(context: Context): File {
        return File(
            context.getExternalFilesDir(Environment.DIRECTORY_MOVIES),
            "EchoCaptures"
        )
    }

    /** List captures newest-first. Blocking; call from a background thread. */
    fun listCaptures(context: Context): List<CaptureInfo> {
        val root = capturesRoot(context)
        if (!root.exists()) return emptyList()
        val dirs = root.listFiles { f -> f.isDirectory && f.name.startsWith("capture_") }
            ?.sortedByDescending { it.name } ?: return emptyList()
        return dirs.mapNotNull { readInfo(it) }
    }

    private fun readInfo(dir: File): CaptureInfo? {
        val meta = try {
            JSONObject(File(dir, "meta.json").readText())
        } catch (e: Exception) {
            null
        }
        val dateMs = meta?.optLong("start_wallclock_ms", 0L) ?: 0L
        // Video duration + frame count from frames.csv (the trusted clock).
        var videoDurationS = meta?.optDouble("duration_s", 0.0) ?: 0.0
        var firstFrameT = Double.NaN
        var lastFrameT = Double.NaN
        try {
            File(dir, "frames.csv").bufferedReader().useLines { lines ->
                var first = true
                for (line in lines) {
                    if (first) {
                        first = false
                        continue // header
                    }
                    val t = line.split(',').getOrNull(1)?.toDoubleOrNull() ?: continue
                    if (firstFrameT.isNaN()) firstFrameT = t
                    lastFrameT = t
                }
            }
            if (!lastFrameT.isNaN() && !firstFrameT.isNaN()) {
                videoDurationS = lastFrameT - firstFrameT
            }
        } catch (ignored: Exception) {
        }
        // Poses: count rows + span.
        var poseCount = 0L
        var firstPoseT = Double.NaN
        var lastPoseT = Double.NaN
        try {
            File(dir, "poses.csv").bufferedReader().useLines { lines ->
                var first = true
                for (line in lines) {
                    if (first) {
                        first = false
                        continue // header
                    }
                    if (line.isBlank()) continue
                    poseCount++
                    val t = line.split(',').getOrNull(0)?.toDoubleOrNull() ?: continue
                    if (firstPoseT.isNaN()) firstPoseT = t
                    lastPoseT = t
                }
            }
        } catch (ignored: Exception) {
        }
        val coverage = if (videoDurationS > 0 && !lastPoseT.isNaN() && !firstPoseT.isNaN()) {
            (lastPoseT - firstPoseT) / videoDurationS * 100.0
        } else {
            0.0
        }
        val size = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
        return CaptureInfo(
            name = dir.name,
            dir = dir,
            dateMs = if (dateMs > 0) dateMs else dir.lastModified(),
            sizeBytes = size,
            videoDurationS = videoDurationS,
            poseCount = poseCount,
            poseCoveragePct = coverage.coerceIn(0.0, 100.0),
        )
    }

    /** Delete a capture folder (all files). Returns true on success. */
    fun delete(capture: CaptureInfo): Boolean {
        return try {
            capture.dir.deleteRecursively()
        } catch (e: Exception) {
            false
        }
    }

    fun formatSize(bytes: Long): String = when {
        bytes >= 1_073_741_824L -> "%.2f GB".format(bytes / 1_073_741_824.0)
        bytes >= 1_048_576L -> "%.1f MB".format(bytes / 1_048_576.0)
        bytes >= 1024L -> "%.0f KB".format(bytes / 1024.0)
        else -> "$bytes B"
    }
}
