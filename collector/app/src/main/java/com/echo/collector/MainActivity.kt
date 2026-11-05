package com.echo.collector

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.echo.collector.databinding.ActivityMainBinding
import org.json.JSONObject
import java.io.File

/**
 * The record screen. One screen, one job.
 *
 * - Big Record / Stop button (pulses while recording).
 * - Live status line: elapsed time, pose count, frame count, tracking
 *   state, active lens.
 * - Lens toggle (Main / Ultrawide). Disabled while recording; the lens is
 *   chosen before a capture starts.
 * - Sensor checklist: which sensors the probe found and will record.
 * - Warning area: the stream watchdog's stale-stream alerts, shown
 *   immediately and named (never silent).
 * - First run: the capability probe runs once in the background and its
 *   summary is shown here; the full manifest lives in Diagnostics.
 *
 * The Activity never touches the camera, ARCore or sensors. It starts and
 * stops [RecorderService] with intents and listens to its broadcasts.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val REQUEST_PERMISSIONS = 1
        private const val PREFS = "echo_collector"
        private const val KEY_PROBED = "capability_probed"
        private const val KEY_LAST_RESULT = "last_result"
    }

    private lateinit var binding: ActivityMainBinding
    private var recording = false
    private var selectedLens = LensHelper.Lens.MAIN
    private var hasUltrawide = false
    /** When the current recording started (local clock). Used by the
     *  poller to show the user time is passing even before the first
     *  status broadcast arrives. */
    private var recordStartMs: Long = 0
    /** When the last ACTION_STATUS broadcast arrived. If the service
     *  stops reporting, the poller says so instead of leaving a frozen
     *  status on screen. */
    private var lastStatusMs: Long = 0
    private val uiHandler = Handler(Looper.getMainLooper())

    /**
     * Direct preview + liveness poller. Runs on the main thread while
     * recording, independent of the status broadcasts — so the preview
     * and the clock keep moving even if a broadcast is missed. (v2.1.1:
     * the preview only updated from broadcasts, and a dead ticker froze
     * the whole screen.)
     */
    private val previewPollRunnable = object : Runnable {
        override fun run() {
            if (!recording) return
            updatePreview()
            val now = SystemClock.elapsedRealtime()
            if (now - lastStatusMs > 2000 && recordStartMs > 0) {
                val elapsed = (now - recordStartMs) / 1000
                val mm = (elapsed / 60).toInt()
                val ss = (elapsed % 60).toInt()
                binding.statusText.text = getString(
                    R.string.status_waiting, mm, ss
                )
            }
            uiHandler.postDelayed(this, 500)
        }
    }

    private fun startPreviewPoller() {
        uiHandler.removeCallbacks(previewPollRunnable)
        uiHandler.post(previewPollRunnable)
    }

    private fun stopPreviewPoller() {
        uiHandler.removeCallbacks(previewPollRunnable)
    }
    private var pulseAnimator: ObjectAnimator? = null
    private var pulseAnimatorY: ObjectAnimator? = null

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                RecorderService.ACTION_STATUS -> {
                    lastStatusMs = SystemClock.elapsedRealtime()
                    val elapsed = intent.getDoubleExtra(RecorderService.EXTRA_ELAPSED_S, 0.0)
                    val frames = intent.getLongExtra(RecorderService.EXTRA_FRAMES, 0)
                    val poses = intent.getLongExtra(RecorderService.EXTRA_POSES, 0)
                    val tracking = intent.getStringExtra(RecorderService.EXTRA_TRACKING) ?: "—"
                    val backend = intent.getStringExtra(RecorderService.EXTRA_BACKEND) ?: "—"
                    val stale = intent.getStringArrayListExtra(RecorderService.EXTRA_STALE)
                        ?: arrayListOf()
                    updateStatus(elapsed, frames, poses, tracking, backend, stale)
                    updatePreview()
                }
                RecorderService.ACTION_WARNING -> {
                    val warning = intent.getStringExtra(RecorderService.EXTRA_WARNING) ?: return
                    showWarning(warning)
                }
                RecorderService.ACTION_FINISHED -> {
                    recording = false
                    stopPreviewPoller()
                    updateRecordButton()
                    val warning = intent.getStringExtra(RecorderService.EXTRA_WARNING)
                    if (warning != null) showWarning(warning)
                    val selfcheck = intent.getStringExtra(RecorderService.EXTRA_SELFCHECK)
                    val dir = intent.getStringExtra(RecorderService.EXTRA_CAPTURE_DIR)
                    showSelfCheck(selfcheck, dir)
                    saveLastResult(selfcheck, dir, warning)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.recordButton.setOnClickListener { onRecordButton() }
        binding.lensMainButton.setOnClickListener { selectLens(LensHelper.Lens.MAIN) }
        binding.lensUltrawideButton.setOnClickListener { selectLens(LensHelper.Lens.ULTRAWIDE) }
        binding.capturesButton.setOnClickListener {
            startActivity(Intent(this, CapturesActivity::class.java))
        }
        binding.diagnosticsButton.setOnClickListener {
            startActivity(Intent(this, DiagnosticsActivity::class.java))
        }
        binding.clearWarningButton.setOnClickListener {
            binding.warningCard.visibility = View.GONE
        }
        selectLens(LensHelper.Lens.MAIN)
        updateRecordButton()
        loadLastResult()

        ensurePermissions()
        loadManifest()
        maybeRunProbe()
    }

    /** One-line summary of the last recording, kept on screen so a failed
     *  recording is visible instead of silently vanishing. */
    private fun loadLastResult() {
        val saved = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY_LAST_RESULT, null)
        if (!saved.isNullOrEmpty()) {
            binding.lastResultText.text = saved
        }
    }

    private fun saveLastResult(selfcheckJson: String?, dir: String?, warning: String?) {
        val summary = try {
            val json = JSONObject(selfcheckJson ?: "{}")
            val streams = json.optJSONObject("streams") ?: JSONObject()
            val poses = streams.optJSONObject("poses")
            val poseCount = poses?.optLong("count", 0) ?: 0
            val poseCov = poses?.optDouble("coverage_pct", 0.0) ?: 0.0
            val overall = json.optString("overall", "?")
            val name = dir?.substringAfterLast('/') ?: ""
            val base = getString(
                R.string.last_result_line, name, poseCount, poseCov, overall
            )
            if (warning != null) "$base\n$warning" else base
        } catch (e: Exception) {
            getString(R.string.last_result_line, dir?.substringAfterLast('/') ?: "", 0, 0.0, "?")
        }
        getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString(KEY_LAST_RESULT, summary).apply()
        binding.lastResultText.text = summary
    }

    /** Live camera preview while recording. The session sends small
     *  thumbnails (~5 fps) through the service; this shows the latest one
     *  in the preview ImageView. Called on every status update (~2x/s).
     *  The old TextureView + Camera2 surface approach is gone: ARCore now
     *  owns the camera, so there is no Camera2 surface to feed. */
    private fun updatePreview() {
        val bmp = RecorderService.latestPreview ?: return
        if (bmp.isRecycled) return
        binding.previewView.setImageBitmap(bmp)
        binding.previewHintText.visibility = View.GONE
    }

    override fun onDestroy() {
        super.onDestroy()
    }

    override fun onResume() {
        super.onResume()
        // If we missed the FINISHED broadcast (the Activity was paused
        // when the recording stopped), the button would stick on
        // "Stopping…". Re-sync with the service's real state.
        if (recording && !RecorderService.isRecording) {
            recording = false
            stopPreviewPoller()
            updateRecordButton()
        }
        if (recording) {
            // Resume the direct preview poller (it stops in onPause).
            startPreviewPoller()
        }
        val filter = IntentFilter().apply {
            addAction(RecorderService.ACTION_STATUS)
            addAction(RecorderService.ACTION_WARNING)
            addAction(RecorderService.ACTION_FINISHED)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(receiver, filter, RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("DEPRECATION")
            registerReceiver(receiver, filter)
        }
    }

    override fun onPause() {
        super.onPause()
        stopPreviewPoller()
        try {
            unregisterReceiver(receiver)
        } catch (ignored: IllegalArgumentException) {
        }
    }

    // ------------------------------------------------------------ permissions
    private fun ensurePermissions() {
        val needed = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) needed.add(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) needed.add(Manifest.permission.POST_NOTIFICATIONS)
        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), REQUEST_PERMISSIONS)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSIONS) {
            // If the camera permission was just granted and we have never
            // probed, run the probe now (it needs the permission to open cameras).
            maybeRunProbe()
        }
    }

    // ------------------------------------------------------------ probe
    private fun loadManifest() {
        try {
            val file = File(filesDir, "capability.json")
            if (!file.exists()) return
            val json = JSONObject(file.readText())
            binding.probeStatusText.text = probeSummary(json)
            updateSensorChecklist(json)
        } catch (ignored: Exception) {
        }
        // Probe already ran in an earlier launch: re-check whether an
        // ultrawide rear camera is exposed, off the UI thread.
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            Thread({
                val ultrawide = try {
                    LensHelper.enumerateBackCameras(this)
                        .any { it.lens == LensHelper.Lens.ULTRAWIDE }
                } catch (e: Exception) {
                    false
                }
                runOnUiThread {
                    hasUltrawide = ultrawide
                    updateLensButtons()
                }
            }, "lens-check").start()
        }
    }

    private fun maybeRunProbe() {
        val prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (prefs.getBoolean(KEY_PROBED, false)) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) return
        binding.probeStatusText.text = getString(R.string.probe_running)
        Thread({
            val manifest = CapabilityProbe.run(this)
            CapabilityProbe.writeToFile(manifest, File(filesDir, "capability.json"))
            // The probe also tells us whether an ultrawide rear camera is
            // actually exposed. This phone exposes only one rear camera,
            // so the Ultrawide button must not pretend otherwise.
            val ultrawide = try {
                LensHelper.enumerateBackCameras(this)
                    .any { it.lens == LensHelper.Lens.ULTRAWIDE }
            } catch (e: Exception) {
                false
            }
            runOnUiThread {
                getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                    .putBoolean(KEY_PROBED, true).apply()
                binding.probeStatusText.text = probeSummary(manifest.toJson())
                updateSensorChecklist(manifest.toJson())
                hasUltrawide = ultrawide
                updateLensButtons()
                Toast.makeText(this, R.string.probe_done, Toast.LENGTH_SHORT).show()
            }
        }, "capability-probe").start()
    }

    private fun updateLensButtons() {
        binding.lensUltrawideButton.isEnabled = hasUltrawide
        binding.lensUltrawideButton.alpha = if (hasUltrawide) 1.0f else 0.35f
        if (!hasUltrawide) {
            binding.lensNoteText.text = getString(R.string.lens_no_ultrawide)
            if (selectedLens == LensHelper.Lens.ULTRAWIDE) {
                selectLens(LensHelper.Lens.MAIN)
            }
        }
    }

    private fun probeSummary(json: JSONObject): String {
        val cams = json.optJSONArray("cameras")?.length() ?: 0
        val camsOpen = (0 until (json.optJSONArray("cameras")?.length() ?: 0)).count { i ->
            json.optJSONArray("cameras")?.optJSONObject(i)?.optBoolean("opens") == true
        }
        val sensors = json.optJSONArray("sensors")?.length() ?: 0
        val arcore = json.optJSONObject("arcore")
        val arOk = arcore?.optBoolean("session_creatable") == true
        return getString(
            R.string.probe_summary, cams, camsOpen, sensors,
            if (arOk) getString(R.string.ok) else getString(R.string.not_ok)
        )
    }

    private fun updateSensorChecklist(json: JSONObject) {
        val sensors = json.optJSONArray("sensors") ?: return
        val lines = StringBuilder()
        for (i in 0 until sensors.length()) {
            val s = sensors.optJSONObject(i) ?: continue
            if (!s.optBoolean("will_record")) continue
            val name = sensorNameZh(s.optString("type_name"))
            lines.append("● ").append(name).append('\n')
        }
        // Barometer is the one that may be absent; say so plainly.
        val hasBaro = (0 until sensors.length()).any { i ->
            sensors.optJSONObject(i)?.optString("type_name") == "barometer"
        }
        if (!hasBaro) lines.append("○ 气压计 · 这台手机没有\n")
        binding.sensorChecklistText.text = lines.toString().trimEnd()
    }

    /** English sensor type name (from the probe) -> Chinese, for display. */
    private fun sensorNameZh(typeName: String): String = when (typeName.lowercase()) {
        "accelerometer" -> "加速度计"
        "gyroscope" -> "陀螺仪"
        "magnetometer", "magnetic_field" -> "磁力计"
        "light" -> "光线"
        "barometer", "pressure" -> "气压计"
        "proximity" -> "距离"
        "gravity" -> "重力"
        "rotation_vector" -> "旋转向量"
        "linear_acceleration" -> "线性加速度"
        else -> typeName
    }

    // ------------------------------------------------------------ recording
    private fun onRecordButton() {
        if (!recording) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED
            ) {
                Toast.makeText(this, R.string.camera_permission_needed, Toast.LENGTH_LONG).show()
                ensurePermissions()
                return
            }
            val intent = Intent(this, RecorderService::class.java).apply {
                action = RecorderService.ACTION_START
                putExtra(RecorderService.EXTRA_LENS, selectedLens.name)
            }
            ContextCompat.startForegroundService(this, intent)
            recording = true
            recordStartMs = SystemClock.elapsedRealtime()
            lastStatusMs = 0
            binding.warningCard.visibility = View.GONE
            updateRecordButton()
            startPulse()
            startPreviewPoller()
        } else {
            val intent = Intent(this, RecorderService::class.java).apply {
                action = RecorderService.ACTION_STOP
            }
            startService(intent)
            // The FINISHED broadcast flips the button back.
            binding.recordButton.isEnabled = false
            binding.recordButton.text = getString(R.string.stopping)
        }
    }

    private fun selectLens(lens: LensHelper.Lens) {
        if (recording) return
        selectedLens = lens
        val mainSelected = lens == LensHelper.Lens.MAIN
        binding.lensMainButton.alpha = if (mainSelected) 1.0f else 0.45f
        binding.lensUltrawideButton.alpha = if (mainSelected) 0.45f else 1.0f
        binding.lensNoteText.text = if (mainSelected) {
            getString(R.string.lens_main_note)
        } else {
            getString(R.string.lens_ultrawide_note)
        }
    }

    private fun updateRecordButton() {
        binding.recordButton.isEnabled = true
        val tint = if (recording) R.color.record_red else R.color.accent_blue
        binding.recordButton.backgroundTintList =
            android.content.res.ColorStateList.valueOf(
                ContextCompat.getColor(this, tint)
            )
        if (recording) {
            binding.recordButton.text = getString(R.string.stop)
            binding.lensMainButton.isEnabled = false
            binding.lensUltrawideButton.isEnabled = false
        } else {
            binding.recordButton.text = getString(R.string.record)
            binding.lensMainButton.isEnabled = true
            binding.lensUltrawideButton.isEnabled = true
            stopPulse()
        }
    }

    private fun updateStatus(
        elapsed: Double, frames: Long, poses: Long, tracking: String, backend: String,
        stale: List<String>
    ) {
        val mm = (elapsed.toInt() / 60)
        val ss = (elapsed.toInt() % 60)
        binding.statusText.text = getString(
            R.string.status_line, mm, ss, poses, frames, tracking, backend
        )
        if (stale.isNotEmpty()) {
            showWarning(getString(R.string.stale_warning, stale.joinToString(", ")))
        }
    }

    private fun showWarning(text: String) {
        binding.warningCard.visibility = View.VISIBLE
        binding.warningText.text = text
    }

    private fun showSelfCheck(selfcheckJson: String?, dir: String?) {
        if (selfcheckJson == null) return
        val message = try {
            val json = JSONObject(selfcheckJson)
            val streams = json.optJSONObject("streams") ?: JSONObject()
            val poses = streams.optJSONObject("poses")
            val poseCov = poses?.optDouble("coverage_pct", 0.0) ?: 0.0
            val poseCount = poses?.optLong("count", 0) ?: 0
            val overall = json.optString("overall", "?")
            val warnings = json.optJSONArray("warnings")
            val warnText = if (warnings != null && warnings.length() > 0) {
                "\n\n" + (0 until warnings.length()).joinToString("\n") { warnings.optString(it) }
            } else ""
            getString(
                R.string.selfcheck_summary,
                dir?.substringAfterLast('/') ?: "",
                poseCount, poseCov, overall
            ) + warnText
        } catch (e: Exception) {
            selfcheckJson
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.selfcheck_title)
            .setMessage(message)
            .setPositiveButton(android.R.string.ok, null)
            .show()
    }

    // ------------------------------------------------------------ animation
    private fun startPulse() {
        stopPulse()
        val sx = ObjectAnimator.ofFloat(binding.recordButton, "scaleX", 1.0f, 1.06f, 1.0f).apply {
            duration = 1000
            repeatCount = ValueAnimator.INFINITE
        }
        val sy = ObjectAnimator.ofFloat(binding.recordButton, "scaleY", 1.0f, 1.06f, 1.0f).apply {
            duration = 1000
            repeatCount = ValueAnimator.INFINITE
        }
        pulseAnimator = sx
        pulseAnimatorY = sy
        sx.start()
        sy.start()
    }

    private fun stopPulse() {
        pulseAnimator?.cancel()
        pulseAnimatorY?.cancel()
        pulseAnimator = null
        pulseAnimatorY = null
        binding.recordButton.scaleX = 1.0f
        binding.recordButton.scaleY = 1.0f
    }
}
