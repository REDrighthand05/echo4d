package com.echo.collector

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Log
import android.util.Size
import kotlin.math.atan

/**
 * Camera lens enumeration and selection.
 *
 * The OnePlus Ace 2 Pro (PJA110) exposes its rear cameras as separate
 * Camera2 devices. We classify them by focal length, not by id, because
 * ids are not stable across devices:
 * - main: the back camera whose focal length is closest to 24 mm
 *   (Sony IMX890, 72 deg horizontal FOV).
 * - ultrawide: the back camera with the shortest focal length
 *   (Sony IMX355, 16 mm, 112 deg FOV).
 *
 * ARCore note: ARCore's shared camera only supports the camera ARCore
 * itself chooses (normally the main). The ultrawide is therefore a
 * video + sensors mode; ARCore poses are not available on it. The caller
 * must surface that clearly (see RecorderSession).
 */
object LensHelper {

    private const val TAG = "LensHelper"

    enum class Lens { MAIN, ULTRAWIDE }

    data class LensInfo(
        val cameraId: String,
        val lens: Lens,
        /** Focal length in mm (first available value). */
        val focalLengthMm: Float,
        /** Horizontal field of view in degrees, derived from focal length
         *  and the physical sensor width. */
        val hFovDeg: Double,
        /** Physical sensor size in mm. */
        val sensorSizeMm: Size?,
        /** Video output sizes the camera can produce (for MediaRecorder). */
        val videoSizes: List<Size>,
    )

    /**
     * Enumerate every back-facing camera and classify it. Returns one entry
     * per back camera, main first. Never throws; a camera whose
     * characteristics cannot be read is skipped and logged.
     */
    fun enumerateBackCameras(context: Context): List<LensInfo> {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val found = mutableListOf<LensInfo>()
        for (id in manager.cameraIdList) {
            try {
                val c = manager.getCameraCharacteristics(id)
                val facing = c.get(CameraCharacteristics.LENS_FACING)
                if (facing != CameraCharacteristics.LENS_FACING_BACK) continue
                val focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    ?.firstOrNull() ?: continue
                val sensorSize = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                val videoSizes = map
                    ?.getOutputSizes(android.media.MediaRecorder::class.java)
                    ?.toList() ?: emptyList()
                val hFov = if (sensorSize != null && focal > 0f) {
                    2.0 * Math.toDegrees(atan(sensorSize.width / 2.0 / focal.toDouble()))
                } else {
                    Double.NaN
                }
                found.add(LensInfo(id, Lens.MAIN, focal, hFov, sensorSize, videoSizes))
            } catch (e: Exception) {
                Log.w(TAG, "cannot read characteristics for camera $id", e)
            }
        }
        if (found.isEmpty()) return emptyList()
        // Shortest focal length = widest lens = ultrawide. The rest is main.
        // With two rear cameras this gives exactly main + ultrawide.
        val sorted = found.sortedBy { it.focalLengthMm }
        val widest = sorted.first()
        val result = found.map {
            if (it.cameraId == widest.cameraId && found.size > 1) {
                it.copy(lens = Lens.ULTRAWIDE)
            } else {
                it.copy(lens = Lens.MAIN)
            }
        }
        return result.sortedBy { if (it.lens == Lens.MAIN) 0 else 1 }
    }

    /** Pick the camera id for the requested lens, or null if absent. */
    fun cameraIdFor(context: Context, lens: Lens): String? =
        enumerateBackCameras(context).firstOrNull { it.lens == lens }?.cameraId

    /** Human-readable horizontal FOV for a focal length + sensor width. */
    fun hFovDeg(focalLengthMm: Float, sensorWidthMm: Float): Double {
        if (focalLengthMm <= 0f || sensorWidthMm <= 0f) return Double.NaN
        return 2.0 * Math.toDegrees(atan(sensorWidthMm / 2.0 / focalLengthMm.toDouble()))
    }
}
