package echo.collector

import android.app.Activity
import android.content.ContentProvider
import android.content.ContentValues
import android.content.Intent
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * One-tap send-back for a capture folder: zip it (stored, no
 * compression pass — the video is already compressed and speed
 * matters more than the CSV bytes) into the app cache and hand it to
 * the system share sheet. The user picks the channel (USB happens by
 * cable anyway; the sheet covers chat apps, mail, cloud drives).
 */
object ShareExport {

    fun zipFolder(folder: File, outFile: File): File {
        ZipOutputStream(FileOutputStream(outFile)).use { zos ->
            zos.setLevel(0)
            folder.walkTopDown().filter { it.isFile }.forEach { f ->
                val rel = f.relativeTo(folder).path.replace('\\', '/')
                zos.putNextEntry(ZipEntry("${folder.name}/$rel"))
                f.inputStream().use { it.copyTo(zos) }
                zos.closeEntry()
            }
        }
        return outFile
    }

    fun share(activity: Activity, folder: File) {
        val zip = File(activity.cacheDir, "${folder.name}.zip")
        val newest = folder.walkTopDown().filter { it.isFile }
            .maxOfOrNull { it.lastModified() } ?: 0L
        if (!zip.exists() || zip.lastModified() < newest) {
            zipFolder(folder, zip)
        }
        val uri = Uri.Builder()
            .scheme("content")
            .authority(ShareProvider.AUTHORITY)
            .appendPath(zip.name)
            .build()
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "application/zip"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, folder.name)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        activity.startActivity(Intent.createChooser(intent, "发送采集文件夹"))
    }
}

/** Minimal read-only provider serving the zips ShareExport builds.
 *  (androidx FileProvider would do, but the app deliberately carries
 *  no androidx dependency.) Only files directly inside the app cache
 *  directory, only by bare file name — no path traversal. */
class ShareProvider : ContentProvider() {

    override fun onCreate(): Boolean = true

    private fun resolve(uri: Uri): File {
        val name = uri.lastPathSegment ?: throw FileNotFoundException("no name")
        require(!name.contains('/') && !name.contains('\\') && name != "..") {
            "bad name"
        }
        val f = File(context!!.cacheDir, name)
        if (!f.exists()) throw FileNotFoundException(name)
        return f
    }

    override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor {
        if (mode != "r") throw FileNotFoundException("read-only")
        return ParcelFileDescriptor.open(resolve(uri),
            ParcelFileDescriptor.MODE_READ_ONLY)
    }

    override fun query(
        uri: Uri, projection: Array<out String>?, selection: String?,
        selectionArgs: Array<out String>?, sortOrder: String?,
    ): Cursor {
        val f = resolve(uri)
        val cursor = MatrixCursor(
            arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE)
        )
        cursor.addRow(arrayOf(f.name, f.length()))
        return cursor
    }

    override fun getType(uri: Uri): String = "application/zip"

    override fun insert(uri: Uri, values: ContentValues?): Uri? =
        throw UnsupportedOperationException("read-only")

    override fun delete(uri: Uri, s: String?, a: Array<out String>?): Int =
        throw UnsupportedOperationException("read-only")

    override fun update(
        uri: Uri, v: ContentValues?, s: String?, a: Array<out String>?,
    ): Int = throw UnsupportedOperationException("read-only")

    companion object {
        const val AUTHORITY = "echo.collector.files"
    }
}
