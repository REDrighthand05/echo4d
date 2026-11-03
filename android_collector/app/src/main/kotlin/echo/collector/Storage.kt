package echo.collector

import android.content.Context
import android.os.Build
import android.os.Environment
import java.io.File

/**
 * Where captures live. Preferred: a shared folder,
 * sdcard/ECHO/captures — visible to Syncthing, which syncs
 * every finished capture to the user's PC on its own (the
 * stable pipe home). That needs the user's one-time all-files
 * access grant; without it the app quietly uses its private
 * external folder, exactly as before, and the in-app direct
 * upload remains the way home.
 */
object Storage {

    fun sharedGranted(): Boolean =
        Build.VERSION.SDK_INT >= 30 &&
            Environment.isExternalStorageManager()

    fun base(context: Context): File =
        if (sharedGranted()) {
            File(Environment.getExternalStorageDirectory(), "ECHO")
        } else {
            context.getExternalFilesDir(null) ?: context.filesDir
        }

    fun capturesDir(context: Context): File =
        File(base(context), "captures").apply { mkdirs() }
}
