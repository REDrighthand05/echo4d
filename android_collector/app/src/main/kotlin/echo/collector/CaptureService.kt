package echo.collector

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import java.io.File

/**
 * Foreground owner of a capture. The recording must survive the
 * screen going off and the user pocketing the phone mid-walk: an
 * Activity alone gets neither guarantee, so the controller lives
 * here, under a camera-type foreground service with a partial wake
 * lock, a persistent notification (with a stop action) and a 1 Hz
 * housekeeping tick (flush rows to disk, enforce the storage floor).
 * The Activity is a thin view on this service's state.
 */
class CaptureService : Service() {

    var controller: CaptureController? = null
        private set
    private var wakeLock: PowerManager.WakeLock? = null
    private val handler = Handler(Looper.getMainLooper())
    private val ticker = object : Runnable {
        override fun run() {
            val c = controller
            if (c != null && c.state == CaptureController.State.RECORDING) {
                c.tick()
                updateNotification()
                handler.postDelayed(this, 1000)
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        val nm = getSystemService(NotificationManager::class.java)
        nm?.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID, "ECHO 采集",
                NotificationManager.IMPORTANCE_LOW
            )
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startCapture()
            ACTION_STOP -> stopCapture()
        }
        return START_NOT_STICKY
    }

    private fun startCapture() {
        if (controller?.state == CaptureController.State.RECORDING) return
        val caps = pendingCaps
        if (caps == null) {
            lastError = "no capabilities handed to the service"
            stopSelf()
            return
        }
        val c = CaptureController(this, caps) { msg -> lastError = msg }
        c.onAutoStop = { f ->
            lastCapture = f
            finishStop()
        }
        latestPreview = null
        c.onPreviewFrame = { bmp -> latestPreview = bmp }
        controller = c
        startForegroundCompat(buildNotification("录制中 00:00"))
        val pm = getSystemService(PowerManager::class.java)
        wakeLock = pm?.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK, "echo:recording"
        )?.apply { acquire(4 * 60 * 60 * 1000L) } // 4 h ceiling
        try {
            c.start()
        } catch (e: Exception) {
            lastError = e.message
            finishStop()
            return
        }
        handler.post(ticker)
    }

    private fun stopCapture() {
        val c = controller ?: return finishStop()
        if (c.state == CaptureController.State.RECORDING) {
            try {
                lastCapture = c.stop()
            } catch (e: Exception) {
                lastError = e.message
            }
        }
        finishStop()
    }

    private fun finishStop() {
        handler.removeCallbacks(ticker)
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun startForegroundCompat(notification: Notification) {
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(
                NOTIF_ID, notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
            )
        } else {
            startForeground(NOTIF_ID, notification)
        }
    }

    private fun buildNotification(text: String): Notification {
        val stopIntent = PendingIntent.getService(
            this, 0,
            Intent(this, CaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val openIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("ECHO 采集")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentIntent(openIntent)
            .addAction(0, "停止", stopIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification() {
        val c = controller ?: return
        val secs = c.elapsedMs / 1000
        val text = "录制中 %02d:%02d · 帧 %d · 位姿 %d".format(
            secs / 60, secs % 60, c.frameCount, c.poseCount
        )
        val nm = getSystemService(NotificationManager::class.java)
        nm?.notify(NOTIF_ID, buildNotification(text))
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        instance = null
        super.onDestroy()
    }

    companion object {
        const val ACTION_START = "echo.collector.START"
        const val ACTION_STOP = "echo.collector.STOP"
        private const val CHANNEL_ID = "echo_capture"
        private const val NOTIF_ID = 42

        @Volatile var instance: CaptureService? = null
        @Volatile var pendingCaps: Capabilities? = null
        @Volatile var lastCapture: File? = null
        @Volatile var lastError: String? = null

        /** Latest preview still from the running backend. */
        @Volatile var latestPreview: android.graphics.Bitmap? = null
    }
}
