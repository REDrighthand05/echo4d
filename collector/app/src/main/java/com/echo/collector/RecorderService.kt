package com.echo.collector

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Foreground service holding one [RecorderSession].
 *
 * Why a service (and not just the Activity):
 * - Recording must survive the screen turning off and the app being
 *   backgrounded. An Activity gets paused and its process can be
 *   throttled; a foreground service cannot.
 * - The partial wake lock keeps the CPU running so the ARCore update loop
 *   and the sensor callbacks keep firing while the screen is off.
 *
 * The service is the only thing that owns the session. The Activity
 * starts/stops it with intents and listens to status broadcasts; it never
 * touches the camera, ARCore or sensors directly.
 *
 * Broadcasts (action [ACTION_STATUS], extras below) go out ~2x/second:
 * - EXTRA_ELAPSED_S, EXTRA_FRAMES, EXTRA_POSES, EXTRA_TRACKING,
 *   EXTRA_LENS, EXTRA_ARCORE_ACTIVE, EXTRA_STALE (StringArrayList).
 * A warning broadcast (action [ACTION_WARNING], EXTRA_WARNING) fires
 * immediately when the watchdog finds a stalled stream.
 * A finished broadcast (action [ACTION_FINISHED]) carries the self-check
 * JSON string (EXTRA_SELFCHECK) and the capture folder path
 * (EXTRA_CAPTURE_DIR).
 */
class RecorderService : Service() {

    companion object {
        const val ACTION_STATUS = "com.echo.collector.STATUS"
        const val ACTION_WARNING = "com.echo.collector.WARNING"
        const val ACTION_FINISHED = "com.echo.collector.FINISHED"
        const val ACTION_START = "com.echo.collector.START"
        const val ACTION_STOP = "com.echo.collector.STOP"
        const val EXTRA_LENS = "lens"
        const val EXTRA_ELAPSED_S = "elapsed_s"
        const val EXTRA_FRAMES = "frames"
        const val EXTRA_POSES = "poses"
        const val EXTRA_TRACKING = "tracking"
        const val EXTRA_ARCORE_ACTIVE = "arcore_active"
        const val EXTRA_STALE = "stale"
        const val EXTRA_WARNING = "warning"
        const val EXTRA_SELFCHECK = "selfcheck"
        const val EXTRA_CAPTURE_DIR = "capture_dir"

        private const val TAG = "RecorderService"
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "echo_capture"
        /** How long stop() may take before we force-finish anyway. */
        private const val STOP_TIMEOUT_MS = 10_000L

        /** True while a recording session is live. Read by the Activity to
         *  recover if it missed the FINISHED broadcast. */
        @Volatile
        var isRecording = false
            private set

        /**
         * Latest preview thumbnail from the recording session (a small
         * grayscale bitmap, ~5 fps). The Activity polls it and shows it
         * in the preview ImageView. Volatile so the recording thread and
         * the UI thread see the same reference; the old bitmap is
         * recycled when replaced.
         */
        @Volatile
        var latestPreview: android.graphics.Bitmap? = null
            private set

        fun updatePreview(bmp: android.graphics.Bitmap) {
            val old = latestPreview
            latestPreview = bmp
            try {
                old?.recycle()
            } catch (ignored: Exception) {
            }
        }

        fun clearPreview() {
            val old = latestPreview
            latestPreview = null
            try {
                old?.recycle()
            } catch (ignored: Exception) {
            }
        }
    }

    private var session: RecorderSession? = null
    private var captureDir: File? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val lensName = intent.getStringExtra(EXTRA_LENS) ?: LensHelper.Lens.MAIN.name
                val lens = try {
                    LensHelper.Lens.valueOf(lensName)
                } catch (e: IllegalArgumentException) {
                    LensHelper.Lens.MAIN
                }
                startRecording(lens)
            }
            ACTION_STOP -> stopRecording()
        }
        // If Android kills us, restart and (there is no session to resume;
        // the user must press Record again). START_STICKY is still the
        // right choice: it keeps the service alive under memory pressure.
        return START_STICKY
    }

    private fun startRecording(lens: LensHelper.Lens) {
        if (session != null) return
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(
            getExternalFilesDir(android.os.Environment.DIRECTORY_MOVIES),
            "EchoCaptures/capture_$stamp"
        )
        captureDir = dir

        // Wake lock first: the CPU must not sleep once we start.
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK, "EchoCollector::Recording"
        ).apply {
            setReferenceCounted(false)
            acquire()
        }

        startForeground(NOTIFICATION_ID, buildNotification("Starting…", 0, 0, "—"))

        isRecording = true
        try {
            session = RecorderSession(
                context = this,
                captureDir = dir,
                lens = lens,
            onStatus = { status ->
                val notification = buildNotification(
                    "Recording ${formatElapsed(status.elapsedS)}",
                    status.poseCount, status.frameCount, status.tracking
                )
                val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                nm.notify(NOTIFICATION_ID, notification)
                val b = Intent(ACTION_STATUS).apply {
                    putExtra(EXTRA_ELAPSED_S, status.elapsedS)
                    putExtra(EXTRA_FRAMES, status.frameCount)
                    putExtra(EXTRA_POSES, status.poseCount)
                    putExtra(EXTRA_TRACKING, status.tracking)
                    putExtra(EXTRA_LENS, status.lens.name)
                    putExtra(EXTRA_ARCORE_ACTIVE, status.arcoreActive)
                    putExtra(EXTRA_CAPTURE_DIR, dir.absolutePath)
                    putStringArrayListExtra(EXTRA_STALE, ArrayList(status.staleStreams))
                }
                sendBroadcast(b)
            },
            onWarning = { warning ->
                Log.w(TAG, warning)
                val b = Intent(ACTION_WARNING).apply {
                    putExtra(EXTRA_WARNING, warning)
                    putExtra(EXTRA_CAPTURE_DIR, dir.absolutePath)
                }
                sendBroadcast(b)
            },
            onPreview = { bmp -> updatePreview(bmp) },
            )
            session?.start()
        } catch (e: Exception) {
            // Recording setup must never crash the app. Log it, tell the
            // user, and shut the service down cleanly.
            Log.e(TAG, "failed to start recording session", e)
            session = null
            try {
                wakeLock?.release()
            } catch (ignored: Exception) {
            }
            wakeLock = null
            val b = Intent(ACTION_WARNING).apply {
                putExtra(EXTRA_WARNING, "Could not start recording: ${e.javaClass.simpleName}: ${e.message}")
                putExtra(EXTRA_CAPTURE_DIR, dir.absolutePath)
            }
            sendBroadcast(b)
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun stopRecording() {
        val s = session ?: run {
            isRecording = false
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return
        }
        session = null
        // stop() is blocking (it tears down camera/ARCore and writes the
        // self-check). Run it off the main thread. If it hangs, a watchdog
        // finishes anyway after STOP_TIMEOUT_MS so the UI never sticks on
        // "Stopping…" — a stuck teardown must not trap the user.
        val finished = java.util.concurrent.atomic.AtomicBoolean(false)
        fun finish(check: StreamWatchdog.SelfCheck?, timedOut: Boolean) {
            if (!finished.compareAndSet(false, true)) return
            isRecording = false
            clearPreview()
            try {
                wakeLock?.release()
            } catch (ignored: Exception) {
            }
            wakeLock = null
            val b = Intent(ACTION_FINISHED).apply {
                val json = check?.toJson()?.toString()
                putExtra(EXTRA_SELFCHECK, json)
                putExtra(EXTRA_CAPTURE_DIR, captureDir?.absolutePath)
                if (timedOut) {
                    putExtra(EXTRA_WARNING, getString(R.string.stop_timeout))
                }
            }
            sendBroadcast(b)
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
        Thread({
            val check = try {
                s.stop()
            } catch (e: Exception) {
                Log.e(TAG, "session stop failed", e)
                null
            }
            finish(check, timedOut = false)
        }, "recorder-stop").start()
        Thread({
            try {
                Thread.sleep(STOP_TIMEOUT_MS)
            } catch (ignored: InterruptedException) {
            }
            finish(null, timedOut = true)
        }, "recorder-stop-watchdog").start()
    }

    private fun buildNotification(
        title: String, poses: Long, frames: Long, tracking: String
    ): Notification {
        createChannel()
        val stopIntent = Intent(this, RecorderService::class.java).apply { action = ACTION_STOP }
        val stopPending = PendingIntent.getService(
            this, 0, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val openIntent = Intent(this, MainActivity::class.java)
        val openPending = PendingIntent.getActivity(
            this, 0, openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText("poses $poses · frames $frames · $tracking")
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentIntent(openPending)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stopPending)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (nm.getNotificationChannel(CHANNEL_ID) != null) return
        val channel = NotificationChannel(
            CHANNEL_ID, "Echo capture", NotificationManager.IMPORTANCE_LOW
        ).apply { description = "Shows recording status while capturing." }
        nm.createNotificationChannel(channel)
    }

    private fun formatElapsed(seconds: Double): String {
        val s = seconds.toInt()
        return "%02d:%02d".format(s / 60, s % 60)
    }

    override fun onDestroy() {
        // If the service is destroyed mid-recording (should not happen with
        // a foreground service + wake lock), stop cleanly so files are closed.
        try {
            session?.stop()
        } catch (ignored: Exception) {
        }
        session = null
        try {
            wakeLock?.release()
        } catch (ignored: Exception) {
        }
        wakeLock = null
        super.onDestroy()
    }
}
