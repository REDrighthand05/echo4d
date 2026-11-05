package com.echo.collector

import android.app.Application
import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Application class. Installs a global crash handler FIRST, so that any
 * uncaught exception (on any thread) writes a full stack trace to a file
 * instead of vanishing. The user can share that file and we can fix the
 * real cause instead of guessing.
 *
 * Crash files land in the app's external files dir
 * (Android/data/com.echo.collector/files/crash-<stamp>.log) and a copy in
 * the internal files dir, so they survive even if external storage is
 * unavailable. The Diagnostics screen lists them.
 */
class EchoCollectorApp : Application() {

    companion object {
        private const val TAG = "EchoCollectorApp"

        fun installCrashHandler(context: Context) {
            val previous = Thread.getDefaultUncaughtExceptionHandler()
            Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
                try {
                    val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                    val sw = StringWriter()
                    throwable.printStackTrace(PrintWriter(sw))
                    val text = buildString {
                        append("EchoCollector crash\n")
                        append("time: ").append(Date().toString()).append('\n')
                        append("device: ").append(Build.MANUFACTURER).append(' ')
                            .append(Build.MODEL).append(" (").append(Build.DEVICE).append(")\n")
                        append("android: ").append(Build.VERSION.RELEASE)
                            .append(" (SDK ").append(Build.VERSION.SDK_INT).append(")\n")
                        append("app: 2.1.0\n")
                        append("thread: ").append(thread.name).append('\n')
                        append("---- stack trace ----\n")
                        append(sw.toString())
                    }
                    for (dir in listOfNotNull(
                        context.getExternalFilesDir(null),
                        context.filesDir
                    )) {
                        try {
                            File(dir, "crash-$stamp.log").writeText(text)
                        } catch (ignored: Exception) {
                        }
                    }
                    Log.e(TAG, "crash written to crash-$stamp.log", throwable)
                } catch (ignored: Exception) {
                }
                previous?.uncaughtException(thread, throwable)
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        installCrashHandler(this)
    }
}
