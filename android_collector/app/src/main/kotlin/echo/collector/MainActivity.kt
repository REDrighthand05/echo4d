package echo.collector

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.io.File

/**
 * One screen, deliberately minimal: the capability line, one big
 * record button, elapsed time, per-stream counts, storage left.
 * Nothing else — capture settings are the capability probe's job, not
 * the user's.
 */
class MainActivity : Activity() {

    private lateinit var capLine: TextView
    private lateinit var statusLine: TextView
    private lateinit var recordButton: Button
    private lateinit var caps: Capabilities
    private var controller: CaptureController? = null
    private val uiHandler = Handler(Looper.getMainLooper())
    private val ticker = object : Runnable {
        override fun run() {
            refreshStatus()
            uiHandler.postDelayed(this, 250)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        caps = Capabilities.probe(this)
        controller = CaptureController(this, caps) { msg ->
            runOnUiThread { statusLine.text = "错误： $msg" }
        }
        writeFirstRunManifest()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 64, 48, 48)
            setBackgroundColor(Color.rgb(16, 18, 22))
        }
        capLine = TextView(this).apply {
            text = caps.summaryLine
            setTextColor(Color.rgb(170, 185, 205))
            textSize = 14f
        }
        root.addView(capLine)
        recordButton = Button(this).apply {
            text = "● 开始录制"
            textSize = 26f
            setOnClickListener { toggle() }
        }
        root.addView(
            recordButton,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 320
            ).apply { topMargin = 64; bottomMargin = 48 }
        )
        statusLine = TextView(this).apply {
            text = "就绪"
            setTextColor(Color.WHITE)
            textSize = 18f
            gravity = Gravity.CENTER
        }
        root.addView(statusLine)
        setContentView(root)

        requestPermissions(
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ),
            1
        )
        uiHandler.post(ticker)
    }

    private fun writeFirstRunManifest() {
        try {
            val f = File(getExternalFilesDir(null), "capabilities.json")
            if (!f.exists()) f.writeText(caps.manifest.render())
        } catch (e: Exception) {
            /* the manifest also travels inside every capture's meta.json */
        }
    }

    private fun toggle() {
        val c = controller ?: return
        if (c.state == CaptureController.State.IDLE) {
            if (checkSelfPermission(Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED
            ) {
                statusLine.text = "需要相机权限"
                return
            }
            try {
                c.start()
                window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                recordButton.text = "■ 停止"
            } catch (e: Exception) {
                statusLine.text = "启动失败： ${e.message}"
            }
        } else {
            try {
                val folder = c.stop()
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                recordButton.text = "● 开始录制"
                statusLine.text = "已保存： ${folder.absolutePath}"
            } catch (e: Exception) {
                statusLine.text = "停止失败： ${e.message}"
            }
        }
    }

    private fun refreshStatus() {
        val c = controller ?: return
        if (c.state != CaptureController.State.RECORDING) return
        val secs = c.elapsedMs / 1000
        val mm = secs / 60
        val ss = secs % 60
        val gb = c.storageLeftBytes() / 1e9
        statusLine.text = "%02d:%02d · 帧 %d · IMU %d · ToF %d · 剩余 %.1f GB".format(
            mm, ss, c.frameCount, c.imuCount, c.tofCount, gb
        )
    }

    override fun onDestroy() {
        uiHandler.removeCallbacks(ticker)
        super.onDestroy()
    }
}
