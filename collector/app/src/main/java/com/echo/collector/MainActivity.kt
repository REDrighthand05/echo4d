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
    }

    private lateinit var binding: ActivityMainBinding
    private var recording = false
    private var selectedLens = LensHelper.Lens.MAIN
    private var pulseAnimator: ObjectAnimator? = null
    private var pulseAnimatorY: ObjectAnimator? = null

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                RecorderService.ACTION_STATUS -> {
                    val elapsed = intent.getDoubleExtra(RecorderService.EXTRA_ELAPSED_S, 0.0)
                    val frames = intent.getLongExtra(RecorderService.EXTRA_FRAMES, 0)
                    val poses = intent.getLongExtra(RecorderService.EXTRA_POSES, 0)
                    val tracking = intent.getStringExtra(RecorderService.EXTRA_TRACKING) ?: "—"
                    val stale = intent.getStringArrayListExtra(RecorderService.EXTRA_STALE)
                        ?: arrayListOf()
                    updateStatus(elapsed, frames, poses, tracking, stale)
                }
                RecorderService.ACTION_WARNING -> {
                    val warning = intent.getStringExtra(RecorderService.EXTRA_WARNING) ?: return
                    showWarning(warning)
                }
                RecorderService.ACTION_FINISHED -> {
                    recording = false
                    updateRecordButton()
                    val selfcheck = intent.getStringExtra(RecorderService.EXTRA_SELFCHECK)
                    val dir = intent.getStringExtra(RecorderService.EXTRA_CAPTURE_DIR)
                    showSelfCheck(selfcheck, dir)
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

        ensurePermissions()
        loadManifest()
        maybeRunProbe()
    }

    override fun onResume() {
        super.onResume()
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
            runOnUiThread {
                getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                    .putBoolean(KEY_PROBED, true).apply()
                binding.probeStatusText.text = probeSummary(manifest.toJson())
                updateSensorChecklist(manifest.toJson())
                Toast.makeText(this, R.string.probe_done, Toast.LENGTH_SHORT).show()
            }
        }, "capability-probe").start()
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
            val name = s.optString("type_name")
            lines.append("● ").append(name).append('\n')
        }
        // Barometer is the one that may be absent; say so plainly.
        val hasBaro = (0 until sensors.length()).any { i ->
            sensors.optJSONObject(i)?.optString("type_name") == "barometer"
        }
        if (!hasBaro) lines.append("○ barometer · not present on this phone\n")
        binding.sensorChecklistText.text = lines.toString().trimEnd()
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
            binding.warningCard.visibility = View.GONE
            updateRecordButton()
            startPulse()
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
        elapsed: Double, frames: Long, poses: Long, tracking: String, stale: List<String>
    ) {
        val mm = (elapsed.toInt() / 60)
        val ss = (elapsed.toInt() % 60)
        binding.statusText.text = getString(
            R.string.status_line, mm, ss, poses, frames, tracking
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
