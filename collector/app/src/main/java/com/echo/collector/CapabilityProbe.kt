package com.echo.collector

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorManager
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Session
import com.google.ar.core.exceptions.UnavailableException
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.EnumSet
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * First-run capability probe.
 *
 * Enumerates EVERYTHING on the phone and TESTS it, so there is never again
 * a "we don't know why the data stopped" moment. The result is a manifest
 * ([CapabilityManifest]) that is:
 * - shown in the Diagnostics screen,
 * - written to capability.json in the app's files dir and into every
 *   capture folder.
 *
 * Camera test: for each camera id we read its characteristics AND actually
 * open it briefly (up to 3 s) and close it again. A camera that lists
 * fine but fails to open is reported with the exact exception message.
 *
 * Sensor test: every sensor from SensorManager.getSensorList(TYPE_ALL) is
 * listed with its static properties. The ones the recorder uses are
 * marked willRecord=true.
 *
 * ARCore test: ArCoreApk.checkAvailability() plus an actual attempt to
 * create a Session (and close it again). Creating the session is the
 * only way to know it really works on this device.
 *
 * This runs on a background thread. It must NOT run while a recording is
 * in progress (it opens cameras). Callers must guard that.
 */
object CapabilityProbe {

    private const val TAG = "CapabilityProbe"
    private const val CAMERA_OPEN_TIMEOUT_MS = 3000L

    data class CameraEntry(
        val id: String,
        val facing: String,
        val focalLengthsMm: List<Float>,
        val sensorSizeMm: String?,
        val outputSizes: List<String>,
        val opens: Boolean,
        val openError: String?,
    )

    data class SensorEntry(
        val type: Int,
        val typeName: String,
        val name: String,
        val vendor: String,
        val maxRange: Float,
        val resolution: Float,
        val powerMa: Float,
        val minDelayUs: Int,
        val willRecord: Boolean,
    )

    data class ArCoreEntry(
        val availability: String,
        val sessionCreatable: Boolean,
        val error: String?,
    )

    data class CapabilityManifest(
        val probeTimeWallclockMs: Long,
        val deviceModel: String,
        val manufacturer: String,
        val androidRelease: String,
        val sdkInt: Int,
        val cameras: List<CameraEntry>,
        val sensors: List<SensorEntry>,
        val arcore: ArCoreEntry,
    ) {
        fun toJson(): JSONObject {
            val root = JSONObject()
            root.put("probe_time_wallclock_ms", probeTimeWallclockMs)
            root.put("device_model", deviceModel)
            root.put("manufacturer", manufacturer)
            root.put("android_release", androidRelease)
            root.put("sdk_int", sdkInt)
            val cams = JSONArray()
            for (c in cameras) {
                val o = JSONObject()
                o.put("id", c.id)
                o.put("facing", c.facing)
                o.put("focal_lengths_mm", JSONArray(c.focalLengthsMm.map { it.toDouble() }))
                o.put("sensor_size_mm", c.sensorSizeMm ?: JSONObject.NULL)
                o.put("output_sizes", JSONArray(c.outputSizes))
                o.put("opens", c.opens)
                o.put("open_error", c.openError ?: JSONObject.NULL)
                cams.put(o)
            }
            root.put("cameras", cams)
            val sens = JSONArray()
            for (s in sensors) {
                val o = JSONObject()
                o.put("type", s.type)
                o.put("type_name", s.typeName)
                o.put("name", s.name)
                o.put("vendor", s.vendor)
                o.put("max_range", s.maxRange.toDouble())
                o.put("resolution", s.resolution.toDouble())
                o.put("power_ma", s.powerMa.toDouble())
                o.put("min_delay_us", s.minDelayUs)
                o.put("will_record", s.willRecord)
                sens.put(o)
            }
            root.put("sensors", sens)
            val ar = JSONObject()
            ar.put("availability", arcore.availability)
            ar.put("session_creatable", arcore.sessionCreatable)
            ar.put("error", arcore.error ?: JSONObject.NULL)
            root.put("arcore", ar)
            return root
        }
    }

    /** Sensor types the recorder actually records. Everything else is listed but not recorded. */
    private val RECORDED_TYPES = setOf(
        Sensor.TYPE_ACCELEROMETER,
        Sensor.TYPE_GYROSCOPE,
        Sensor.TYPE_MAGNETIC_FIELD,
        Sensor.TYPE_LIGHT,
        Sensor.TYPE_PRESSURE,
    )

    fun sensorTypeName(type: Int): String = when (type) {
        Sensor.TYPE_ACCELEROMETER -> "accelerometer"
        Sensor.TYPE_GYROSCOPE -> "gyroscope"
        Sensor.TYPE_MAGNETIC_FIELD -> "magnetometer"
        Sensor.TYPE_LIGHT -> "light"
        Sensor.TYPE_PRESSURE -> "barometer"
        Sensor.TYPE_PROXIMITY -> "proximity"
        Sensor.TYPE_GRAVITY -> "gravity"
        Sensor.TYPE_LINEAR_ACCELERATION -> "linear_acceleration"
        Sensor.TYPE_ROTATION_VECTOR -> "rotation_vector"
        Sensor.TYPE_AMBIENT_TEMPERATURE -> "ambient_temperature"
        Sensor.TYPE_RELATIVE_HUMIDITY -> "relative_humidity"
        Sensor.TYPE_STEP_COUNTER -> "step_counter"
        Sensor.TYPE_STEP_DETECTOR -> "step_detector"
        else -> "type_$type"
    }

    /**
     * Run the full probe. Blocking; call from a background thread.
     * Returns the manifest. Never throws; failures are recorded per item.
     */
    fun run(context: Context): CapabilityManifest {
        val cameras = probeCameras(context)
        val sensors = probeSensors(context)
        val arcore = probeArCore(context)
        return CapabilityManifest(
            probeTimeWallclockMs = System.currentTimeMillis(),
            deviceModel = android.os.Build.MODEL ?: "unknown",
            manufacturer = android.os.Build.MANUFACTURER ?: "unknown",
            androidRelease = android.os.Build.VERSION.RELEASE ?: "unknown",
            sdkInt = android.os.Build.VERSION.SDK_INT,
            cameras = cameras,
            sensors = sensors,
            arcore = arcore,
        )
    }

    private fun probeCameras(context: Context): List<CameraEntry> {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val hasPermission = ContextCompat.checkSelfPermission(
            context, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        val out = mutableListOf<CameraEntry>()
        val ids = try {
            manager.cameraIdList
        } catch (e: CameraAccessException) {
            Log.w(TAG, "cannot list cameras", e)
            return listOf(
                CameraEntry("?", "unknown", emptyList(), null, emptyList(), false, e.message)
            )
        }
        for (id in ids) {
            var facing = "unknown"
            var focal = emptyList<Float>()
            var sensorSize: String? = null
            var sizes = emptyList<String>()
            try {
                val c = manager.getCameraCharacteristics(id)
                facing = when (c.get(CameraCharacteristics.LENS_FACING)) {
                    CameraCharacteristics.LENS_FACING_BACK -> "back"
                    CameraCharacteristics.LENS_FACING_FRONT -> "front"
                    CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
                    else -> "unknown"
                }
                focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList()
                    ?: emptyList()
                val ss = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                sensorSize = ss?.let { "${it.width}x${it.height} mm" }
                val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                sizes = map?.getOutputSizes(android.graphics.ImageFormat.JPEG)
                    ?.map { "${it.width}x${it.height}" }?.sorted() ?: emptyList()
            } catch (e: Exception) {
                out.add(CameraEntry(id, facing, focal, sensorSize, sizes, false, "characteristics: ${e.message}"))
                continue
            }
            // Actually open the camera briefly. This is the test that matters.
            var opens = false
            var openError: String? = null
            if (!hasPermission) {
                openError = "CAMERA permission not granted"
            } else {
                val result = tryOpenCamera(manager, id)
                opens = result.first
                openError = result.second
            }
            out.add(CameraEntry(id, facing, focal, sensorSize, sizes, opens, openError))
        }
        return out
    }

    /**
     * Open [cameraId], wait for onOpened (or an error) up to the timeout,
     * then close it. Returns (opened, errorMessage).
     */
    private fun tryOpenCamera(manager: CameraManager, cameraId: String): Pair<Boolean, String?> {
        val thread = HandlerThread("probe-camera-$cameraId").also { it.start() }
        val handler = Handler(thread.looper)
        val latch = CountDownLatch(1)
        var opened: CameraDevice? = null
        var error: String? = null
        try {
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    opened = device
                    latch.countDown()
                }

                override fun onDisconnected(device: CameraDevice) {
                    error = "disconnected while opening"
                    latch.countDown()
                }

                override fun onError(device: CameraDevice, code: Int) {
                    error = "open error code=$code"
                    latch.countDown()
                }
            }, handler)
            val ok = latch.await(CAMERA_OPEN_TIMEOUT_MS, TimeUnit.MILLISECONDS)
            if (!ok) error = "open timed out after ${CAMERA_OPEN_TIMEOUT_MS}ms"
        } catch (e: CameraAccessException) {
            error = "CameraAccessException: ${e.message}"
        } catch (e: SecurityException) {
            error = "SecurityException: ${e.message}"
        } catch (e: IllegalArgumentException) {
            error = "IllegalArgumentException: ${e.message}"
        } finally {
            try {
                opened?.close()
            } catch (ignored: Exception) {
            }
            thread.quitSafely()
        }
        return Pair(opened != null && error == null, error)
    }

    private fun probeSensors(context: Context): List<SensorEntry> {
        val manager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val list = try {
            manager.getSensorList(Sensor.TYPE_ALL)
        } catch (e: Exception) {
            Log.w(TAG, "cannot list sensors", e)
            return emptyList()
        }
        return list.map { s ->
            SensorEntry(
                type = s.type,
                typeName = sensorTypeName(s.type),
                name = s.name ?: "unknown",
                vendor = s.vendor ?: "unknown",
                maxRange = s.maximumRange,
                resolution = s.resolution,
                powerMa = s.power,
                minDelayUs = s.minDelay,
                willRecord = s.type in RECORDED_TYPES,
            )
        }.sortedBy { it.typeName }
    }

    private fun probeArCore(context: Context): ArCoreEntry {
        val availability = try {
            ArCoreApk.getInstance().checkAvailability(context).name
        } catch (e: Exception) {
            return ArCoreEntry("check failed: ${e.message}", false, e.message)
        }
        if (availability != ArCoreApk.Availability.SUPPORTED_INSTALLED.name) {
            return ArCoreEntry(availability, false, "ARCore not installed / not supported")
        }
        // Actually create a session. This is the only proof it works.
        var session: Session? = null
        return try {
            session = Session(context, EnumSet.of(Session.Feature.SHARED_CAMERA))
            ArCoreEntry(availability, true, null)
        } catch (e: UnavailableException) {
            ArCoreEntry(availability, false, "Session creation failed: ${e.message}")
        } catch (e: Exception) {
            ArCoreEntry(availability, false, "Session creation failed: ${e.message}")
        } finally {
            try {
                session?.close()
            } catch (ignored: Exception) {
            }
        }
    }

    /** Write the manifest JSON to [file]. */
    fun writeToFile(manifest: CapabilityManifest, file: File) {
        file.parentFile?.mkdirs()
        file.writeText(manifest.toJson().toString(2))
    }
}
