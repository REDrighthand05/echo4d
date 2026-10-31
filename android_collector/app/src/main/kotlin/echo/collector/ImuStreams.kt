package echo.collector

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import echo.shared.Convert

/**
 * IMU + extras on the shared elapsedRealtime clock.
 *
 * imu.csv rows are emitted on ROTATION-VECTOR events (the platform's
 * fused orientation product, the analogue of the BNO's on-chip fusion):
 * each row carries that event's quaternion plus the most recent
 * accelerometer and gyroscope samples (latest-sample hold — the three
 * sensors free-run at their own rates; this is the standard way to get
 * one coherent stream out of SensorManager, and it is documented here
 * rather than hidden). Accel/gyro vectors pass through unchanged:
 * Android's device frame IS the format's device frame (see
 * Convert.kt). Gyro/accel are kept as the raw (uncalibrated where
 * offered) sensors report them.
 *
 * Extras go to sidecar CSVs via the same StampClock discipline:
 * magnetometer, barometer, GPS (Location.elapsedRealtimeNanos is in
 * the same clock domain).
 */
class ImuStreams(
    context: Context,
    private val sink: Sink,
) : SensorEventListener, LocationListener {

    interface Sink {
        fun onImuRow(tNs: Long, quatWxyz: DoubleArray, accel: DoubleArray, gyro: DoubleArray)
        fun onMag(tNs: Long, v: DoubleArray)
        fun onBaro(tNs: Long, hPa: Double)
        fun onGps(tNs: Long, lat: Double, lon: Double, alt: Double, acc: Double)
    }

    private val sm = context.getSystemService(SensorManager::class.java)
    private val lm = context.getSystemService(LocationManager::class.java)
    private var accel: Sensor? = null
    private var gyro: Sensor? = null
    private var rotVec: Sensor? = null
    private var mag: Sensor? = null
    private var baro: Sensor? = null

    @Volatile private var lastAccel = DoubleArray(3)
    @Volatile private var lastGyro = DoubleArray(3)
    @Volatile private var haveAccel = false
    @Volatile private var haveGyro = false
    var imuRows = 0
        private set
    var gpsEnabled = false
        private set

    fun start(handler: Handler) {
        accel = sm?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyro = sm?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotVec = sm?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        mag = sm?.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        baro = sm?.getDefaultSensor(Sensor.TYPE_PRESSURE)
        // FASTEST + zero batching latency: true per-event timestamps.
        accel?.let { sm?.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, 0, handler) }
        gyro?.let { sm?.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, 0, handler) }
        rotVec?.let { sm?.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST, 0, handler) }
        mag?.let { sm?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME, 0, handler) }
        baro?.let { sm?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME, 0, handler) }
        try {
            if (lm?.isProviderEnabled(LocationManager.GPS_PROVIDER) == true) {
                lm.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER, 1000L, 0f, this, handler.looper
                )
                gpsEnabled = true
            }
        } catch (e: SecurityException) {
            gpsEnabled = false // permission denied: GPS sidecar just stays absent
        }
    }

    fun stop() {
        sm?.unregisterListener(this)
        try { lm?.removeUpdates(this) } catch (e: Exception) { /* already off */ }
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAccel = doubleArrayOf(
                    event.values[0].toDouble(), event.values[1].toDouble(),
                    event.values[2].toDouble()
                )
                haveAccel = true
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGyro = doubleArrayOf(
                    event.values[0].toDouble(), event.values[1].toDouble(),
                    event.values[2].toDouble()
                )
                haveGyro = true
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                if (haveAccel && haveGyro) {
                    sink.onImuRow(
                        event.timestamp,
                        Convert.rotationVectorToWxyz(event.values),
                        lastAccel, lastGyro
                    )
                    imuRows++
                }
            }
            Sensor.TYPE_MAGNETIC_FIELD -> sink.onMag(
                event.timestamp,
                doubleArrayOf(
                    event.values[0].toDouble(), event.values[1].toDouble(),
                    event.values[2].toDouble()
                )
            )
            Sensor.TYPE_PRESSURE -> sink.onBaro(event.timestamp, event.values[0].toDouble())
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* not recorded */ }

    override fun onLocationChanged(location: Location) {
        sink.onGps(
            location.elapsedRealtimeNanos,
            location.latitude, location.longitude,
            if (location.hasAltitude()) location.altitude else Double.NaN,
            if (location.hasAccuracy()) location.accuracy.toDouble() else Double.NaN
        )
    }

    @Deprecated("Deprecated in API 29; LocationListener still requires it pre-31")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) { }
    override fun onProviderEnabled(provider: String) { }
    override fun onProviderDisabled(provider: String) { }
}
