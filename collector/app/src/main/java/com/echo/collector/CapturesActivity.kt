package com.echo.collector

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.echo.collector.databinding.ActivityCapturesBinding
import com.echo.collector.databinding.ItemCaptureBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Recording management: a simple list of past captures.
 *
 * Each row shows the capture name, date, size, video duration, pose count
 * and pose coverage %. A delete button removes the whole capture folder
 * (after confirmation). Stats come from the capture files themselves, so
 * the list is always truthful — a capture whose poses died at 6 s shows
 * ~7% coverage here, not a green tick.
 */
class CapturesActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCapturesBinding
    private val captures = mutableListOf<CaptureStore.CaptureInfo>()
    private lateinit var adapter: CaptureAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCapturesBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        adapter = CaptureAdapter(captures) { capture -> confirmDelete(capture) }
        binding.recyclerView.layoutManager = LinearLayoutManager(this)
        binding.recyclerView.adapter = adapter
        loadCaptures()
    }

    override fun onResume() {
        super.onResume()
        loadCaptures()
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    private fun loadCaptures() {
        Thread({
            val list = CaptureStore.listCaptures(this)
            runOnUiThread {
                captures.clear()
                captures.addAll(list)
                adapter.notifyDataSetChanged()
                binding.emptyText.visibility = if (list.isEmpty()) View.VISIBLE else View.GONE
            }
        }, "load-captures").start()
    }

    private fun confirmDelete(capture: CaptureStore.CaptureInfo) {
        AlertDialog.Builder(this)
            .setTitle(R.string.delete_title)
            .setMessage(getString(R.string.delete_message, capture.name))
            .setPositiveButton(R.string.delete) { _, _ ->
                Thread({
                    val ok = CaptureStore.delete(capture)
                    runOnUiThread {
                        if (ok) {
                            Toast.makeText(this, R.string.deleted, Toast.LENGTH_SHORT).show()
                            loadCaptures()
                        } else {
                            Toast.makeText(this, R.string.delete_failed, Toast.LENGTH_SHORT).show()
                        }
                    }
                }, "delete-capture").start()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private class CaptureAdapter(
        private val items: List<CaptureStore.CaptureInfo>,
        private val onDelete: (CaptureStore.CaptureInfo) -> Unit,
    ) : RecyclerView.Adapter<CaptureAdapter.Holder>() {

        private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())

        class Holder(val binding: ItemCaptureBinding) : RecyclerView.ViewHolder(binding.root)

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder {
            val binding = ItemCaptureBinding.inflate(
                LayoutInflater.from(parent.context), parent, false
            )
            return Holder(binding)
        }

        override fun getItemCount(): Int = items.size

        override fun onBindViewHolder(holder: Holder, position: Int) {
            val c = items[position]
            val b = holder.binding
            b.nameText.text = c.name
            b.dateText.text = dateFormat.format(Date(c.dateMs))
            val duration = "%d:%02d".format(
                (c.videoDurationS.toInt() / 60), (c.videoDurationS.toInt() % 60)
            )
            b.statsText.text = b.root.context.getString(
                R.string.capture_stats,
                duration,
                CaptureStore.formatSize(c.sizeBytes),
                c.poseCount
            )
            // Coverage colour: green >= 90%, amber >= 50%, red below.
            val color = when {
                c.poseCoveragePct >= 90.0 -> 0xFF4CAF50.toInt()
                c.poseCoveragePct >= 50.0 -> 0xFFFFC107.toInt()
                else -> 0xFFF44336.toInt()
            }
            b.coverageText.setTextColor(color)
            b.coverageText.text = b.root.context.getString(
                R.string.coverage_pct, c.poseCoveragePct
            )
            b.deleteButton.setOnClickListener { onDelete(c) }
        }
    }
}
