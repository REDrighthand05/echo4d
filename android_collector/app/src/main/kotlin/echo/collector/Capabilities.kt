package echo.collector

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import com.google.ar.core.ArCoreApk
import echo.shared.JsonObj
import kotlin.math.atan

/** What this phone actually has. Probed once at startup, shown as the
 *  one-line summary on the capture screen, and embedded in full into
 *  every capture's meta.json — the manifest is how we learn what the
 *  user's specific OnePlus exposes (depth camera? rates? lens FOV?). */
class Capabilities private constructor(
    val summaryLine: String,
    val manifest: JsonObj,
    val chosenCameraId: String,
    val chosenFovHDeg: Double,
    val chosenSensorOrientation: Int,
    val camera2Depth16: Boolean,
    val arcoreSupported: Boolean,
    val arcoreDepthSupported: Boolean,
) {
    companion object {
        fun probe(context: Context): Capabilities {
            val sm = context.getSystemService(SensorManager::class.java)
            val cm = context.getSystemService(CameraManager::class.java)

            // --- sensors -------------------------------------------
            val sensorJson = ArrayList<Any>()
            var hasAccel = false; var hasGyro = false; var hasRotVec = false
            var hasMag = false; var hasBaro = false
            val all = sm?.getSensorList(Sensor.TYPE_ALL) ?: emptyList()
            for (s in all) {
                when (s.type) {
                    Sensor.TYPE_ACCELEROMETER -> hasAccel = true
                    Sensor.TYPE_GYROSCOPE -> hasGyro = true
                    Sensor.TYPE_ROTATION_VECTOR -> hasRotVec = true
                    Sensor.TYPE_MAGNETIC_FIELD -> hasMag = true
                    Sensor.TYPE_PRESSURE -> hasBaro = true
                }
                sensorJson.add(
                    JsonObj()
                        .put("type", s.type)
                        .put("name", s.name)
                        .put("vendor", s.vendor)
                        .put("max_rate_hz",
                            if (s.minDelay > 0) 1_000_000.0 / s.minDelay else null)
                        .put("resolution", s.resolution)
                )
            }

            // --- cameras -------------------------------------------
            val camJson = ArrayList<Any>()
            var chosenId: String? = null
            var chosenFov = -1.0
            var chosenOrientation = 0
            var chosenDepth = false
            for (id in cm?.cameraIdList ?: emptyArray()) {
                val ch = cm.getCameraCharacteristics(id)
                val facing = ch.get(CameraCharacteristics.LENS_FACING)
                val focals = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                val phys = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                val orient = ch.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
                val caps = ch.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
                    ?.toList() ?: emptyList()
                val depth = caps.contains(
                    CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT)
                val tsSource = ch.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
                var fovH: Double? = null
                if (focals != null && focals.isNotEmpty() && phys != null) {
                    // widest lens of this camera = shortest focal length
                    val f = focals.min()
                    fovH = Math.toDegrees(2.0 * atan((phys.width / 2.0) / f))
                }
                camJson.add(
                    JsonObj()
                        .put("id", id)
                        .put("facing", facing)
                        .put("focal_mm", focals?.map { it.toDouble() })
                        .put("sensor_mm",
                            phys?.let { listOf(it.width.toDouble(), it.height.toDouble()) })
                        .put("fov_h_deg", fovH)
                        .put("sensor_orientation_deg", orient)
                        .put("timestamp_source",
                            if (tsSource == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME)
                                "REALTIME" else "UNKNOWN")
                        .put("depth16_capable", depth)
                )
                if (facing == CameraCharacteristics.LENS_FACING_BACK &&
                    fovH != null && fovH > chosenFov
                ) {
                    chosenFov = fovH; chosenId = id
                    chosenOrientation = orient; chosenDepth = depth
                }
            }

            // --- ARCore ---------------------------------------------
            val arcoreAvail = try {
                ArCoreApk.getInstance().checkAvailability(context)
            } catch (e: Exception) {
                ArCoreApk.Availability.UNSUPPORTED_DEVICE_NOT_CAPABLE
            }
            val arcoreSupported = arcoreAvail == ArCoreApk.Availability.SUPPORTED_INSTALLED ||
                arcoreAvail == ArCoreApk.Availability.SUPPORTED_NOT_INSTALLED ||
                arcoreAvail == ArCoreApk.Availability.SUPPORTED_APK_TOO_OLD
            // Depth support can be probed without resuming a session:
            // constructing a Session does not take the camera.
            var arcoreDepth = false
            if (arcoreSupported) {
                try {
                    val probe = com.google.ar.core.Session(context)
                    arcoreDepth = probe.isDepthModeSupported(
                        com.google.ar.core.Config.DepthMode.AUTOMATIC)
                    probe.close()
                } catch (e: Exception) {
                    arcoreDepth = false
                }
            }

            val depthText = when {
                chosenDepth -> "深度: Camera2 DEPTH16"
                arcoreDepth -> "深度: ARCore"
                arcoreSupported -> "深度: ARCore位姿(无深度)"
                else -> "深度: 无"
            }
            val summary = "${Build.MANUFACTURER} ${Build.MODEL} · " +
                "IMU ${if (hasAccel && hasGyro && hasRotVec) "✓" else "缺"} · " +
                "$depthText · 广角 ${if (chosenFov > 0) "%.0f°".format(chosenFov) else "?"}"

            val manifest = JsonObj()
                .put("device", JsonObj()
                    .put("manufacturer", Build.MANUFACTURER)
                    .put("model", Build.MODEL)
                    .put("device", Build.DEVICE)
                    .put("android_release", Build.VERSION.RELEASE)
                    .put("sdk_int", Build.VERSION.SDK_INT))
                .put("sensors", sensorJson)
                .put("cameras", camJson)
                .put("imu_complete", hasAccel && hasGyro && hasRotVec)
                .put("extras", JsonObj()
                    .put("magnetometer", hasMag)
                    .put("barometer", hasBaro)
                    .put("gps", true))
                .put("depth", JsonObj()
                    .put("camera2_depth16", chosenDepth)
                    .put("arcore_availability", arcoreAvail.name)
                    .put("arcore_supported", arcoreSupported)
                    .put("arcore_depth_supported", arcoreDepth))
                .put("chosen_camera", chosenId)
            return Capabilities(
                summary, manifest, chosenId ?: "0",
                if (chosenFov > 0) chosenFov else 0.0,
                chosenOrientation, chosenDepth, arcoreSupported, arcoreDepth
            )
        }
    }
}
