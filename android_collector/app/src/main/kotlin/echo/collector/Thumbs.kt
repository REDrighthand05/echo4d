package echo.collector

import android.graphics.Bitmap
import android.media.Image

/**
 * Tiny preview thumbnails from a YUV_420_888 image: strided
 * subsample straight to RGB pixels, no intermediate full-size
 * copy. Used by both backends so the UI can show a live-ish
 * preview (~5 Hz stills) while a backend owns the camera —
 * ARCore cannot share its camera with a preview surface, and
 * the Camera2 recorder surface must not be disturbed either.
 */
object Thumbs {

    fun fromYuv(image: Image, targetW: Int): Bitmap? {
        return try {
            val w = image.width
            val h = image.height
            if (w <= 0 || h <= 0) return null
            val step = maxOf(1, w / targetW)
            val outW = w / step
            val outH = h / step
            val yPlane = image.planes[0]
            val uPlane = image.planes[1]
            val vPlane = image.planes[2]
            val yBuf = yPlane.buffer
            val uBuf = uPlane.buffer
            val vBuf = vPlane.buffer
            val pixels = IntArray(outW * outH)
            var idx = 0
            for (row in 0 until outH) {
                val yRow = row * step
                val cRow = yRow / 2
                for (col in 0 until outW) {
                    val x = col * step
                    val yv = yBuf.get(
                        yRow * yPlane.rowStride + x * yPlane.pixelStride
                    ).toInt() and 0xFF
                    val cx = x / 2
                    val uv = uBuf.get(
                        cRow * uPlane.rowStride + cx * uPlane.pixelStride
                    ).toInt() and 0xFF
                    val vv = vBuf.get(
                        cRow * vPlane.rowStride + cx * vPlane.pixelStride
                    ).toInt() and 0xFF
                    val yy = (yv - 16).coerceAtLeast(0) * 1.164f
                    val r = (yy + 1.596f * (vv - 128)).toInt()
                    val g = (yy - 0.813f * (vv - 128) -
                        0.391f * (uv - 128)).toInt()
                    val b = (yy + 2.018f * (uv - 128)).toInt()
                    pixels[idx++] = (0xFF shl 24) or
                        (r.coerceIn(0, 255) shl 16) or
                        (g.coerceIn(0, 255) shl 8) or
                        b.coerceIn(0, 255)
                }
            }
            Bitmap.createBitmap(pixels, outW, outH, Bitmap.Config.ARGB_8888)
        } catch (e: Exception) {
            null
        }
    }
}
