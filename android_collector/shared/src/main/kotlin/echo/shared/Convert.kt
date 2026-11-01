package echo.shared

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Android -> ECHO-format conversions. The design, in full:
 *
 * CLOCKS. Android is the first collection end whose sensors natively
 * share one clock: Camera2's SENSOR_TIMESTAMP and SensorEvent.timestamp
 * are both nanoseconds in the elapsedRealtime domain
 * (SystemClock.elapsedRealtimeNanos) on devices that report
 * INFO_TIMESTAMP_SOURCE_REALTIME — the app records the source it found
 * in meta.json, and refuses to mix domains silently (an UNKNOWN-source
 * camera still uses SENSOR_TIMESTAMP, which is then the best common
 * domain available; the caveat is written into meta). The recorder
 * takes ONE origin (elapsedRealtimeNanos at record start) and every
 * stream stamps t_s = (ts_ns − origin_ns)/1e9 from the EVENT's own
 * timestamp — camera frames from SENSOR_TIMESTAMP, never from arrival
 * time. Compare the tethered head, which had to stamp on arrival and
 * keep a sidecar of MCU ticks: here the sidecar is unnecessary because
 * there is only one clock.
 *
 * DEVICE FRAME. The format's device frame is "x right, y up, z back
 * toward the operator (screen facing the operator)". Android's sensor
 * frame is x toward the right edge, y toward the top edge, z out of
 * the screen toward the operator. These are the same physical axes, so
 * accelerometer/gyro vectors pass through UNCHANGED.
 *
 * WORLD FRAME + QUATERNION. Android's rotation vector reports the
 * orientation as a quaternion in (x, y, z, w) order giving the
 * world-from-device rotation into Android's ENU world frame
 * (x ~east, y ~north, z up). The format wants scalar-first (w,x,y,z)
 * world-from-device into "the IMU's own world frame, fixed at boot" —
 * which is exactly what we store: the Android ENU frame is this
 * capture's world frame, and registering it onto a reconstruction is
 * the importer's documented job (register_imu_frame). The conversion
 * is therefore a pure component reorder, tested below.
 *
 * CAMERA CALIBRATION. cam_from_device maps format device-frame
 * vectors into the OpenCV camera frame (x right, y down, z forward):
 * v_cam = R v_dev, R's rows are the camera axes written in device
 * coordinates. The rear camera looks along −z_device, so z_cam =
 * (0,0,−1). The sensor's raw image right/down axes lie in the device
 * xy-plane, rotated by SENSOR_ORIENTATION θ: Android defines θ as the
 * clockwise rotation the raw image needs to appear upright in the
 * device's natural orientation, from which x_cam = (cosθ, sinθ, 0)
 * and y_cam = (sinθ, −cosθ, 0) (derivation in the unit tests; the
 * θ=0 case reproduces the format's reference mounting
 * diag(1, −1, −1) exactly).
 */
object Convert {

    /** Rotation-vector values (x,y,z[,w]) -> format quaternion (w,x,y,z). */
    fun rotationVectorToWxyz(values: FloatArray): DoubleArray {
        val x = values[0].toDouble()
        val y = values[1].toDouble()
        val z = values[2].toDouble()
        val w = if (values.size >= 4) values[3].toDouble()
        else {
            val s = 1.0 - x * x - y * y - z * z
            if (s > 0) sqrt(s) else 0.0
        }
        val n = sqrt(w * w + x * x + y * y + z * z)
        return if (n > 0) doubleArrayOf(w / n, x / n, y / n, z / n)
        else doubleArrayOf(1.0, 0.0, 0.0, 0.0)
    }

    /** 3x3 cam_from_device for a rear camera with the given
     *  SENSOR_ORIENTATION in degrees (0/90/180/270). */
    fun camFromDevice(sensorOrientationDeg: Int): Array<DoubleArray> {
        val th = Math.toRadians(sensorOrientationDeg.toDouble())
        val c = cos(th)
        val s = sin(th)
        return arrayOf(
            doubleArrayOf(c, s, 0.0),   // x_cam: image right
            doubleArrayOf(s, -c, 0.0),  // y_cam: image down
            doubleArrayOf(0.0, 0.0, -1.0) // z_cam: into the scene
        )
    }
}

/**
 * Downsample an Android DEPTH16 image to the format's 8x8 ToF zones.
 *
 * DEPTH16 packs a 13-bit depth in millimetres (bits 0..12) and a
 * 3-bit confidence (bits 13..15); depth 0 means no measurement. The
 * format's zones tile the imager's field of view uniformly in
 * tan(angle) space; for a pinhole depth camera, equal pixel blocks
 * ARE equal tan-space cells (u = f·tanθ), so the image is cut into an
 * 8x8 grid of equal pixel blocks. A zone's value is the MEDIAN of its
 * valid samples (a robust centre, insensitive to edge outliers and
 * mixed pixels); a zone with no valid sample is TOF_INVALID (−1).
 * Row 0 of the image is the top of the view, matching the format's
 * row-major-from-top-left zone order.
 */
object DepthDownsample {

    fun downsample(depth: ShortArray, width: Int, height: Int): IntArray {
        require(depth.size == width * height) { "depth buffer size mismatch" }
        val out = IntArray(TOF_GRID * TOF_GRID) { TOF_INVALID }
        val cellW = width / TOF_GRID
        val cellH = height / TOF_GRID
        val samples = ArrayList<Int>(cellW * cellH)
        for (r in 0 until TOF_GRID) {
            for (c in 0 until TOF_GRID) {
                samples.clear()
                val y0 = r * cellH
                val x0 = c * cellW
                // Stride 2 in both axes: a zone still sees hundreds of
                // samples on real depth cameras; keeps CPU bounded on
                // full-res depth while changing the median negligibly.
                var y = y0
                while (y < y0 + cellH) {
                    var x = x0
                    val rowBase = y * width
                    while (x < x0 + cellW) {
                        val d = depth[rowBase + x].toInt() and 0x1FFF
                        if (d > 0) samples.add(d)
                        x += 2
                    }
                    y += 2
                }
                if (samples.isNotEmpty()) {
                    samples.sort()
                    out[r * TOF_GRID + c] = samples[samples.size / 2]
                }
            }
        }
        return out
    }
}
