package echo.collector

import android.app.Activity
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * The pipe home: POST a capture zip straight to the receiver
 * running on the user's PC (an HttpListener script Muse
 * installed there). Two addresses are tried in order — the
 * PC's home-LAN IP first (same WiFi: full speed), then its
 * Tailscale IP — so the button works wherever the phone can
 * reach the PC, with no VPN app required on the phone. The
 * receiver answers with the sha256 it computed, so "sent"
 * here means verified arrived, not merely dispatched.
 * (The always-on pipe is Syncthing; this is the manual one.)
 */
object NetUpload {

    private val ENDPOINTS = listOf(
        "http://192.168.71.250:8377/upload",
        "http://100.114.229.10:8377/upload",
    )
    private const val TOKEN = "49f6ff2494d13cb7d51e1e7d8fa57fc5"

    /** Runs the upload on a worker thread. Callbacks also arrive
     *  on that thread — the Activity hops to the UI thread. */
    fun upload(
        activity: Activity,
        folder: File,
        onProgress: (Int) -> Unit,
        onDone: (String?) -> Unit,
    ) {
        Thread {
            val zip = File(activity.cacheDir, "${folder.name}.zip")
            try {
                val newest = folder.walkTopDown().filter { it.isFile }
                    .maxOfOrNull { it.lastModified() } ?: 0L
                if (!zip.exists() || zip.lastModified() < newest) {
                    ShareExport.zipFolder(folder, zip)
                }
            } catch (e: Exception) {
                onDone("打包失败： ${e.message}")
                return@Thread
            }
            var lastError: String? = null
            for (endpoint in ENDPOINTS) {
                try {
                    postOnce(endpoint, zip, onProgress)
                    onDone(null)
                    return@Thread
                } catch (e: Exception) {
                    lastError = e.message
                }
            }
            onDone(
                "连不上电脑（$lastError）——" +
                    "确认手机和电脑连着同一个 WiFi、电脑开着")
        }.start()
    }

    private fun postOnce(
        endpoint: String,
        zip: File,
        onProgress: (Int) -> Unit,
    ) {
        val conn = URL(endpoint).openConnection() as HttpURLConnection
        conn.requestMethod = "POST"
        conn.doOutput = true
        conn.connectTimeout = 6_000
        conn.readTimeout = 60_000
        conn.setRequestProperty("X-Echo-Token", TOKEN)
        conn.setRequestProperty("X-Echo-Name", zip.name)
        conn.setFixedLengthStreamingMode(zip.length())
        val total = zip.length()
        var sent = 0L
        var lastPct = -1
        zip.inputStream().use { ins ->
            conn.outputStream.use { outs ->
                val buf = ByteArray(256 * 1024)
                while (true) {
                    val n = ins.read(buf)
                    if (n < 0) break
                    outs.write(buf, 0, n)
                    sent += n
                    val pct = if (total > 0) {
                        (sent * 100 / total).toInt()
                    } else 100
                    if (pct != lastPct) {
                        lastPct = pct
                        onProgress(pct)
                    }
                }
            }
        }
        val code = conn.responseCode
        if (code !in 200..299) throw Exception("电脑返回 $code")
    }
}
