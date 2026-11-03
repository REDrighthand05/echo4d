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
 *  user's specific OnePlus exposes (depth camera? rates? lens FOV?).
 *
 *  Lens choice (v0.2): candidates are every back-facing LOGICAL
 *  camera plus every PHYSICAL camera inside a logical multi-camera
 *  (CameraCharacteristics.physicalCameraIds). Phones like the
 *  OnePlus 12 expose the ultrawide only as a physical camera of the
 *  main logical device; probing logical ids alone picked the 72°
 *  main lens and never saw the ~114° ultrawide. The widest candidate
 *  wins; the recorder opens the logical parent and targets the
 *  physical lens through its output configurations, falling back to
 *  the plain logical stream if the device refuses. */
class Capabilities private constructor(
    val summaryLine: String,
    val manifest: JsonObj,
    val chosenCameraId: String,
    val chosenPhysicalId: String?,
    val chosenFovHDeg: Double,
    val chosenSensorOrientation: Int,
    val camera2Depth16: Boolean,
    val arcoreSupported: Boolean,
    val arcoreDepthSupported: Boolean,
) {
    companion object {
        private fun fovH(ch: CameraCharacteristics): Double? {
            val focals = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val phys = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            if (focals == null || focals.isEmpty() || phys == null) return null
            val f = focals.min()
            return Math.toDegrees(2.0 * atan((phys.width / 2.0) / f))
        }

        private fun hasDepth(ch: CameraCharacteristics): Boolean =
            ch.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
                ?.contains(
                    CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT
                ) == true

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
            data class Candidate(
                val logicalId: String,
                val physicalId: String?,
                val fov: Double,
                val orientation: Int,
                val depth: Boolean,
            )

            val camJson = ArrayList<Any>()
            var best: Candidate? = null
            fun consider(c: Candidate) {
                if (best == null || c.fov > best!!.fov) best = c
            }
            for (id in cm?.cameraIdList ?: emptyArray()) {
                val ch = cm.getCameraCharacteristics(id)
                val facing = ch.get(CameraCharacteristics.LENS_FACING)
                val focals = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                val phys = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                val orient = ch.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
                val tsSource = ch.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
                val fov = fovH(ch)
                val depth = hasDepth(ch)

                // Physical cameras inside this logical device.
                val physJson = ArrayList<Any>()
                if (facing == CameraCharacteristics.LENS_FACING_BACK &&
                    Build.VERSION.SDK_INT >= 28
                ) {
                    val physicalIds = try {
                        ch.physicalCameraIds
                    } catch (e: Exception) {
                        emptySet<String>()
                    }
                    for (pid in physicalIds) {
                        try {
                            val pch = cm.getCameraCharacteristics(pid)
                            val pfov = fovH(pch)
                            val pfocals = pch.get(
                                CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                            val pphys = pch.get(
                                CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                            physJson.add(
                                JsonObj()
                                    .put("id", pid)
                                    .put("focal_mm", pfocals?.map { it.toDouble() })
                                    .put("sensor_mm", pphys?.let {
                                        listOf(it.width.toDouble(), it.height.toDouble())
                                    })
                                    .put("fov_h_deg", pfov)
                                    .put("depth16_capable", hasDepth(pch))
                            )
                            if (pfov != null) {
                                consider(
                                    Candidate(
                                        id, pid, pfov,
                                        pch.get(CameraCharacteristics
                                            .SENSOR_ORIENTATION) ?: orient,
                                        hasDepth(pch)
                                    )
                                )
                            }
                        } catch (e: Exception) {
                            physJson.add(JsonObj().put("id", pid)
                                .put("error", e.message))
                        }
                    }
                }
                camJson.add(
                    JsonObj()
                        .put("id", id)
                        .put("facing", facing)
                        .put("focal_mm", focals?.map { it.toDouble() })
                        .put("sensor_mm",
                            phys?.let { listOf(it.width.toDouble(), it.height.toDouble()) })
                        .put("fov_h_deg", fov)
                        .put("sensor_orientation_deg", orient)
                        .put("timestamp_source",
                            if (tsSource == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME)
                                "REALTIME" else "UNKNOWN")
                        .put("depth16_capable", depth)
                        .put("physical_cameras", physJson)
                )
                if (facing == CameraCharacteristics.LENS_FACING_BACK && fov != null) {
                    consider(Candidate(id, null, fov, orient, depth))
                }
            }
            val chosen = best
            val chosenId = chosen?.logicalId ?: "0"
            val chosenPhysical = chosen?.physicalId
            val chosenFov = chosen?.fov ?: -1.0
            val chosenOrientation = chosen?.orientation ?: 0
            val chosenDepth = chosen?.depth ?: false

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
            val lensText = if (chosenPhysical != null) "物理镜头$chosenPhysical" else "逻辑镜头"
            val summary = "${Build.MANUFACTURER} ${Build.MODEL} · " +
                "IMU ${if (hasAccel && hasGyro && hasRotVec) "✓" else "缺"} · " +
                "$depthText · 广角 ${if (chosenFov > 0) "%.0f°".format(chosenFov) else "?"} ($lensText)"

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
                .put("chosen_physical_camera", chosenPhysical)
            return Capabilities(
                summary, manifest, chosenId, chosenPhysical,
                if (chosenFov > 0) chosenFov else 0.0,
                chosenOrientation, chosenDepth, arcoreSupported, arcoreDepth
            )
        }
    }
}
