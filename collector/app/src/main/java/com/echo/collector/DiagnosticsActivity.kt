package com.echo.collector

import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.echo.collector.databinding.ActivityDiagnosticsBinding
import org.json.JSONObject
import java.io.File

/**
 * Diagnostics screen: the capability probe manifest, in plain language.
 *
 * Shows, for this exact phone:
 * - every camera: facing, focal lengths, sensor size, output sizes, and
 *   whether it actually OPENS (with the failure reason if it does not);
 * - every sensor: type, name, vendor, range, resolution, power, and
 *   whether the recorder will use it;
 * - ARCore: availability and whether a session can really be created.
 *
 * "Re-run probe" re-runs the whole thing (it opens each camera briefly)
 * and overwrites capability.json. It is disabled while a recording is
 * in progress, because the probe opens cameras.
 */
class DiagnosticsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityDiagnosticsBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDiagnosticsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        binding.rerunButton.setOnClickListener { rerunProbe() }
        showManifest(loadManifest())
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    private fun loadManifest(): JSONObject? {
        return try {
            val file = File(filesDir, "capability.json")
            if (!file.exists()) return null
            JSONObject(file.readText())
        } catch (e: Exception) {
            null
        }
    }

    private fun showManifest(json: JSONObject?) {
        if (json == null) {
            binding.manifestText.text = getString(R.string.no_manifest)
            return
        }
        val out = StringBuilder()
        out.append(getString(R.string.diag_device,
            json.optString("device_model"),
            json.optString("manufacturer"),
            json.optString("android_release"),
            json.optInt("sdk_int")
        )).append("\n\n")

        out.append(getString(R.string.diag_cameras)).append("\n")
        val cams = json.optJSONArray("cameras")
        if (cams == null || cams.length() == 0) {
            out.append("  —\n")
        } else {
            for (i in 0 until cams.length()) {
                val c = cams.optJSONObject(i) ?: continue
                val opens = c.optBoolean("opens")
                val focal = (0 until (c.optJSONArray("focal_lengths_mm")?.length() ?: 0))
                    .joinToString(", ") { c.optJSONArray("focal_lengths_mm")?.optDouble(it).toString() }
                out.append("  • id ${c.optString("id")} (${c.optString("facing")})")
                    .append("  f=$focal mm")
                    .append("  sensor=${c.optString("sensor_size_mm")}")
                    .append("\n    opens: ")
                    .append(if (opens) "YES" else "NO")
                val err = c.optString("open_error")
                if (!opens && err.isNotEmpty()) out.append(" — $err")
                out.append("\n")
            }
        }

        out.append("\n").append(getString(R.string.diag_sensors)).append("\n")
        val sensors = json.optJSONArray("sensors")
        if (sensors == null || sensors.length() == 0) {
            out.append("  —\n")
        } else {
            for (i in 0 until sensors.length()) {
                val s = sensors.optJSONObject(i) ?: continue
                out.append("  • ${s.optString("type_name")} — ${s.optString("name")}")
                    .append(" (${s.optString("vendor")})")
                    .append("\n    range=${s.optDouble("max_range")} res=${s.optDouble("resolution")} power=${s.optDouble("power_ma")}mA")
                    .append("  record=${if (s.optBoolean("will_record")) "YES" else "no"}")
                    .append("\n")
            }
        }

        out.append("\n").append(getString(R.string.diag_arcore)).append("\n")
        val ar = json.optJSONObject("arcore")
        if (ar == null) {
            out.append("  —\n")
        } else {
            out.append("  availability: ${ar.optString("availability")}\n")
            out.append("  session creatable: ${if (ar.optBoolean("session_creatable")) "YES" else "NO"}\n")
            val err = ar.optString("error")
            if (err.isNotEmpty() && err != "null") out.append("  error: $err\n")
        }
        binding.manifestText.text = out.toString()
    }

    private fun rerunProbe() {
        binding.rerunButton.isEnabled = false
        binding.manifestText.text = getString(R.string.probe_running)
        Thread({
            val manifest = CapabilityProbe.run(this)
            CapabilityProbe.writeToFile(manifest, File(filesDir, "capability.json"))
            runOnUiThread {
                binding.rerunButton.isEnabled = true
                showManifest(manifest.toJson())
                Toast.makeText(this, R.string.probe_done, Toast.LENGTH_SHORT).show()
            }
        }, "capability-probe").start()
    }
}
