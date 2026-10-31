package echo.shared

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt

class ConvertTest {

    @Test
    fun rotationVectorReordersToScalarFirst() {
        // 90 degrees about x: xyzw = (sin45, 0, 0, cos45)
        val s = Math.sin(Math.PI / 4).toFloat()
        val c = Math.cos(Math.PI / 4).toFloat()
        val q = Convert.rotationVectorToWxyz(floatArrayOf(s, 0f, 0f, c))
        // inputs are Floats, so the honest delta is float epsilon scale
        assertArrayEquals(doubleArrayOf(c.toDouble(), s.toDouble(), 0.0, 0.0), q, 1e-6)
    }

    @Test
    fun rotationVectorThreeComponentForm() {
        // No w stored: w = sqrt(1 - x^2 - y^2 - z^2), still unit norm.
        val q = Convert.rotationVectorToWxyz(floatArrayOf(0.1f, -0.2f, 0.3f))
        val n = sqrt(q.sumOf { it * it })
        assertEquals(1.0, n, 1e-9)
        assertEquals(sqrt(1.0 - 0.01 - 0.04 - 0.09), q[0], 1e-6)
    }

    @Test
    fun camFromDeviceZeroMatchesReferenceMounting() {
        // The format's reference mounting (sensor_record.py sim +
        // tethered recorder): device -> camera = diag(1, -1, -1).
        val r = Convert.camFromDevice(0)
        val expected = arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(0.0, -1.0, 0.0),
            doubleArrayOf(0.0, 0.0, -1.0)
        )
        for (i in 0..2) assertArrayEquals(expected[i], r[i], 1e-12)
    }

    @Test
    fun camFromDeviceNinety() {
        // θ=90: raw image right points along device +y (the raw
        // landscape image's right edge sits at the top of a portrait
        // phone), raw image down along device +x.
        val r = Convert.camFromDevice(90)
        assertArrayEquals(doubleArrayOf(0.0, 1.0, 0.0), r[0], 1e-9)
        assertArrayEquals(doubleArrayOf(1.0, 0.0, 0.0), r[1], 1e-9)
        assertArrayEquals(doubleArrayOf(0.0, 0.0, -1.0), r[2], 1e-9)
    }

    @Test
    fun camFromDeviceIsAlwaysARotation() {
        for (deg in listOf(0, 90, 180, 270)) {
            val r = Convert.camFromDevice(deg)
            // orthonormal rows
            for (i in 0..2) {
                val n = sqrt(r[i].sumOf { it * it })
                assertEquals(1.0, n, 1e-9)
            }
            val cross = doubleArrayOf(
                r[0][1] * r[1][2] - r[0][2] * r[1][1],
                r[0][2] * r[1][0] - r[0][0] * r[1][2],
                r[0][0] * r[1][1] - r[0][1] * r[1][0]
            )
            assertArrayEquals(r[2], cross, 1e-9) // x cross y = z => det +1
        }
    }

    @Test
    fun gravitySanityFlatDevice() {
        // Device lying flat, screen up, identity orientation: the
        // accelerometer measures +9.81 along device +z (out of the
        // screen). World-from-device identity must carry that to
        // world +z (up) — i.e. the format's up derivation sees
        // gravity exactly where ENU says it is.
        val q = Convert.rotationVectorToWxyz(floatArrayOf(0f, 0f, 0f, 1f))
        assertArrayEquals(doubleArrayOf(1.0, 0.0, 0.0, 0.0), q, 1e-12)
        val accelDev = doubleArrayOf(0.0, 0.0, 9.81)
        // identity rotation: world accel == device accel
        assertTrue(abs(accelDev[2] - 9.81) < 1e-9)
    }
}

class StampClockTest {
    @Test
    fun convertsNanosecondsFromOrigin() {
        val clock = StampClock(1_000_000_000L)
        assertEquals(0.0, clock.stamp("imu", 1_000_000_000L), 1e-12)
        assertEquals(0.01, clock.stamp("imu", 1_010_000_000L), 1e-12)
        assertEquals(1.5, clock.stamp("imu", 2_500_000_000L), 1e-12)
    }

    @Test
    fun forcesStrictlyIncreasingPerStream() {
        val clock = StampClock(0L)
        val a = clock.stamp("imu", 1_000_000L)
        val b = clock.stamp("imu", 1_000_000L) // same event stamp again
        assertTrue(b > a)
        assertEquals(a + 1e-6, b, 1e-12)
        // other streams are independent
        assertEquals(0.001, clock.stamp("tof", 1_000_000L), 1e-12)
    }
}

class DepthDownsampleTest {
    private fun depthImage(w: Int, h: Int, fill: (Int, Int) -> Int): ShortArray {
        val buf = ShortArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val d = fill(x, y)
            buf[y * w + x] = if (d <= 0) 0 else (d or (7 shl 13)).toShort()
        }
        return buf
    }

    @Test
    fun recoversConstantZones() {
        // 80x80 image: zone (r,c) filled with 1000 + r*8 + c mm.
        val img = depthImage(80, 80) { x, y -> 1000 + (y / 10) * 8 + (x / 10) }
        val zones = DepthDownsample.downsample(img, 80, 80)
        for (r in 0..7) for (c in 0..7) {
            assertEquals(1000 + r * 8 + c, zones[r * 8 + c])
        }
    }

    @Test
    fun invalidZoneBecomesMinusOne() {
        val img = depthImage(80, 80) { x, y -> if (x < 10 && y < 10) 0 else 500 }
        val zones = DepthDownsample.downsample(img, 80, 80)
        assertEquals(TOF_INVALID, zones[0])
        assertEquals(500, zones[1])
    }

    @Test
    fun confidenceBitsAreNotDepth() {
        // A zone full of depth 8191 would be a bogus reading if the
        // confidence bits leaked in; use a small depth with max
        // confidence and expect the small depth back.
        val img = depthImage(16, 16) { _, _ -> 120 }
        val zones = DepthDownsample.downsample(img, 16, 16)
        assertTrue(zones.all { it == 120 })
    }
}

class FormatTest {
    @Test
    fun headersMatchPythonContract() {
        assertEquals("frame,t_s,file", FRAMES_HEADER.joinToString(","))
        assertEquals("t_s,qw,qx,qy,qz,ax,ay,az,gx,gy,gz", IMU_HEADER.joinToString(","))
        assertEquals(65, TOF_HEADER.size)
        assertEquals("z0", TOF_HEADER[1])
        assertEquals("z63", TOF_HEADER[64])
    }

    @Test
    fun floatFormattingMatchesPython() {
        // Python: f"{x:.6f}"
        assertEquals("0.250000", fmt(0.25))
        assertEquals("-1.500000", fmt(-1.5))
        assertEquals("159.496000", fmt(159.496))
    }

    @Test
    fun writerEmitsParseableFolder() {
        val dir = createTempDir("echo_fmt_test")
        try {
            val w = CaptureWriter(dir)
            w.addFrame(1, 0.0, "frame_00001.jpg")
            w.addFrame(2, 0.5, "frame_00002.jpg")
            w.addImu(0.0, doubleArrayOf(1.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 9.81), doubleArrayOf(0.0, 0.0, 0.0))
            w.addTof(0.0, IntArray(64) { 1000 + it })
            w.writeStreams()
            val frames = java.io.File(dir, "frames.csv").readLines()
            assertEquals("frame,t_s,file", frames[0])
            assertEquals("1,0.000000,frame_00001.jpg", frames[1])
            val tof = java.io.File(dir, "tof.csv").readLines()
            assertEquals(2, tof.size)
            assertTrue(tof[1].startsWith("0.000000,1000,1001,"))
        } finally {
            dir.deleteRecursively()
        }
    }
}
