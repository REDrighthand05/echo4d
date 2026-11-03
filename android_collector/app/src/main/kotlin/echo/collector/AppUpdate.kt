package echo.collector

import android.app.Activity
import android.content.Intent
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * The collector's own hot update, mirroring the desktop app:
 * check a plain static relay first (GitHub Pages on the
 * dedicated echo4d-update repo), fall back to a CDN mirror of
 * the same files, and never depend on the GitHub API, whose
 * unauthenticated rate limit is shared per egress IP and kept
 * locking the user out. An offer is honoured only when its
 * sha256 matches the downloaded APK; installation goes through
 * the system installer (one tap), the strongest hand-off
 * Android allows a sideloaded app.
 */
object AppUpdate {

    data class Offer(
        val version: String,
        val code: Int,
        val apkUrl: String,
        val sha256: String,
        val notes: String,
    )

    private val INDEX_URLS = listOf(
        "https://redrighthand05.github.io/echo4d-update/collector/index.json",
        "https://cdn.jsdelivr.net/gh/REDrighthand05/echo4d-update@main/collector/index.json",
    )

    /** Blocking; call from a worker thread. Null = no offer
     *  (up to date, or every route failed quietly). */
    fun fetchOffer(currentCode: Int): Offer? {
        for (indexUrl in INDEX_URLS) {
            try {
                val text = httpText(indexUrl) ?: continue
                val root = JSONObject(text)
                val versions = root.getJSONArray("versions")
                var best: Offer? = null
                for (i in 0 until versions.length()) {
                    val v = versions.getJSONObject(i)
                    val code = v.getInt("code")
                    if (code <= currentCode) continue
                    val base = indexUrl.substringBeforeLast('/') + "/"
                    val offer = Offer(
                        version = v.getString("version"),
                        code = code,
                        apkUrl = base + v.getString("apk"),
                        sha256 = v.getString("sha256"),
                        notes = v.optString("notes"),
                    )
                    if (best == null || code > best.code) best = offer
                }
                if (best != null) return best
            } catch (e: Exception) {
                // try the next route
            }
        }
        return null
    }

    /** Blocking download + hash check on a worker thread. */
    fun download(
        activity: Activity,
        offer: Offer,
        onProgress: (Int) -> Unit,
        onReady: (File?, String?) -> Unit,
    ) {
        Thread {
            try {
                val out = File(
                    activity.cacheDir,
                    "EchoCollector-${offer.version}.apk")
                val conn = URL(offer.apkUrl).openConnection()
                    as HttpURLConnection
                conn.connectTimeout = 10_000
                conn.readTimeout = 60_000
                val total = conn.contentLengthLong
                var done = 0L
                var lastPct = -1
                conn.inputStream.use { ins ->
                    out.outputStream().use { outs ->
                        val buf = ByteArray(256 * 1024)
                        while (true) {
                            val n = ins.read(buf)
                            if (n < 0) break
                            outs.write(buf, 0, n)
                            done += n
                            val pct = if (total > 0) {
                                (done * 100 / total).toInt()
                            } else -1
                            if (pct != lastPct && pct >= 0) {
                                lastPct = pct
                                onProgress(pct)
                            }
                        }
                    }
                }
                val digest = MessageDigest.getInstance("SHA-256")
                out.inputStream().use { ins ->
                    val buf = ByteArray(256 * 1024)
                    while (true) {
                        val n = ins.read(buf)
                        if (n < 0) break
                        digest.update(buf, 0, n)
                    }
                }
                val hex = digest.digest().joinToString("") {
                    "%02x".format(it)
                }
                if (!hex.equals(offer.sha256, ignoreCase = true)) {
                    out.delete()
                    onReady(null, "下载校验没通过，已删掉")
                    return@Thread
                }
                onReady(out, null)
            } catch (e: Exception) {
                onReady(null, "下载失败： ${e.message}")
            }
        }.start()
    }

    /** Hand the verified APK to the system installer. */
    fun install(activity: Activity, apk: File) {
        val uri = Uri.parse(
            "content://echo.collector.files/${apk.name}")
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(
                uri, "application/vnd.android.package-archive")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        activity.startActivity(intent)
    }

    private fun httpText(url: String): String? {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 8_000
        conn.readTimeout = 15_000
        if (conn.responseCode !in 200..299) return null
        return conn.inputStream.bufferedReader().readText()
    }
}
