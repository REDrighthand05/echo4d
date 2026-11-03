package echo.collector

import android.content.Context
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import echo.shared.DepthDownsample
import echo.shared.JsonObj
import java.io.File
import java.util.EnumSet

/**
 * ARCore backend — chosen whenever ARCore is supported and the
 * Camera2 path has no measured depth (decided by the controller).
 * ARCore owns the camera in this mode (it cannot share it with a
 * Camera2 session), so this backend produces everything: video (CPU
 * images encoded with MediaCodec), per-frame timestamps
 * (frame.timestamp, the same elapsedRealtime nanosecond domain as
 * SensorEvent), 6DoF poses with their tracking state (poses.csv,
 * ARCore world frame — metres, +Y up, right-handed) and depth
 * downsampled to the 8x8 ToF grid when the device offers it.
 *
 * frames.csv honesty: a frame is stamped only after it has actually
 * been queued to the encoder — a dropped codec input must not invent
 * a frames.csv row the video does not contain. stop() returns the
 * number of samples the muxer actually wrote, so the controller can
 * reconcile exactly like the Camera2 path.
 */
class ArCoreRecorder(
    private val context: Context,
    private val folder: File,
    private val onFrameStamp: (Long) -> Unit,
    private val onPose: (Long, Pose, String) -> Unit,
    private val onDepth: (Long, IntArray) -> Unit,
    private val onPreview: (android.graphics.Bitmap) -> Unit,
) {
    @Volatile private var running = false
    private var thread: Thread? = null
    private var session: Session? = null
    var note: String = "not started"
        private set
    var depthActive: Boolean = false
        private set
    private var encodedFrames = 0
    private var intrinsics: JsonObj? = null
    private var preparedW = 0
    private var preparedH = 0

    // Bring-up handshake: the loop thread performs EGL + session
    // setup (the GL context must live on the thread that calls
    // update()), then reports the verdict through this latch so
    // the controller can fall back BEFORE committing.
    private val bringUpLatch = java.util.concurrent.CountDownLatch(1)
    @Volatile private var bringUpError: String? = null
    private var errorSink: ((String) -> Unit)? = null
    private var eglDisplay: android.opengl.EGLDisplay? = null
    private var eglContext: android.opengl.EGLContext? = null
    private var eglSurface: android.opengl.EGLSurface? = null
    private var cameraTexId = 0

    fun intrinsicsJson(): JsonObj? = intrinsics

    /**
     * Starts the loop thread and waits for its bring-up verdict.
     * Returns null when the session is live, else the reason —
     * the controller then falls back to Camera2 for this capture.
     *
     * Two device lessons are baked in here:
     *  - The config ladder walks from strictest to loosest,
     *    because some devices return an EMPTY config list when a
     *    depth sensor is required. Depth is a preference, never
     *    a gate.
     *  - ARCore delivers NO frames until the session has a
     *    camera texture name: without one, update() returns
     *    frames whose timestamp stays 0 forever (the v0.3
     *    failure on the user's phone — session up, camera mute).
     *    A 1x1 EGL pbuffer context exists here only to own that
     *    texture; the pixels we record come from CPU images.
     */
    fun prepareBlocking(
        onError: (String) -> Unit, timeoutMs: Long = 6_000,
    ): String? {
        start(onError)
        val done = try {
            bringUpLatch.await(timeoutMs, java.util.concurrent.TimeUnit.MILLISECONDS)
        } catch (e: InterruptedException) {
            false
        }
        return when {
            !done -> "arcore bring-up timed out"
            else -> bringUpError
        }
    }

    private fun bringUp() {
        initEglTexture()
        val sess = Session(context)
        sess.setCameraTextureName(cameraTexId)
        val config = Config(sess)
        // BLOCKING, not LATEST_CAMERA_IMAGE: the v0.8 field run
        // showed LATEST spinning 44,844 updates for 44 distinct
        // frames — ARCore handed back its cached frame at ~1.7
        // new frames a second on this phone. BLOCKING makes
        // update() wait for each genuinely new frame; the loop
        // counters in the session note will show whether the
        // delivery rate follows.
        config.updateMode = Config.UpdateMode.BLOCKING
        val depthSupported =
            sess.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
        val rungs: List<Pair<String, (CameraConfigFilter) -> Unit>> =
            listOf(
                "fps30+depth" to { f: CameraConfigFilter ->
                    f.setTargetFps(EnumSet.of(
                        CameraConfig.TargetFps.TARGET_FPS_30))
                    if (depthSupported) {
                        f.setDepthSensorUsage(EnumSet.of(
                            CameraConfig.DepthSensorUsage
                                .REQUIRE_AND_USE))
                    }
                },
                "fps30" to { f: CameraConfigFilter ->
                    f.setTargetFps(EnumSet.of(
                        CameraConfig.TargetFps.TARGET_FPS_30))
                },
                "any" to { _: CameraConfigFilter -> }
            )
        var chosen: CameraConfig? = null
        var rungName = ""
        for ((name, apply) in rungs) {
            val filter = CameraConfigFilter(sess)
            apply(filter)
            val list = sess.getSupportedCameraConfigs(filter)
            if (list.isNotEmpty()) {
                chosen = list.maxByOrNull {
                    it.imageSize.width * it.imageSize.height
                }
                rungName = name
                break
            }
        }
        if (chosen == null) {
            try { sess.close() } catch (e: Exception) { /* done */ }
            throw IllegalStateException(
                "no ARCore camera config (every filter rung empty)")
        }
        depthActive = depthSupported && rungName == "fps30+depth"
        if (depthActive) config.depthMode = Config.DepthMode.AUTOMATIC
        sess.cameraConfig = chosen
        sess.configure(config)
        sess.resume()
        session = sess
        preparedW = chosen.imageSize.width
        preparedH = chosen.imageSize.height
        note = "arcore up (config rung=$rungName, depth=$depthActive, " +
            "${preparedW}x$preparedH)"
    }

    private fun initEglTexture() {
        val display = android.opengl.EGL14.eglGetDisplay(
            android.opengl.EGL14.EGL_DEFAULT_DISPLAY)
        check(display != android.opengl.EGL14.EGL_NO_DISPLAY) {
            "no EGL display"
        }
        val version = IntArray(2)
        check(android.opengl.EGL14.eglInitialize(
            display, version, 0, version, 1)) { "eglInitialize failed" }
        val configAttribs = intArrayOf(
            android.opengl.EGL14.EGL_RED_SIZE, 8,
            android.opengl.EGL14.EGL_GREEN_SIZE, 8,
            android.opengl.EGL14.EGL_BLUE_SIZE, 8,
            android.opengl.EGL14.EGL_ALPHA_SIZE, 8,
            android.opengl.EGL14.EGL_SURFACE_TYPE,
            android.opengl.EGL14.EGL_PBUFFER_BIT,
            android.opengl.EGL14.EGL_RENDERABLE_TYPE,
            android.opengl.EGL14.EGL_OPENGL_ES2_BIT,
            android.opengl.EGL14.EGL_NONE)
        val configs = arrayOfNulls<android.opengl.EGLConfig>(1)
        val numConfigs = IntArray(1)
        check(android.opengl.EGL14.eglChooseConfig(
            display, configAttribs, 0, configs, 0, 1, numConfigs, 0) &&
            numConfigs[0] > 0) { "eglChooseConfig failed" }
        val ctxAttribs = intArrayOf(
            android.opengl.EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
            android.opengl.EGL14.EGL_NONE)
        val ctx = android.opengl.EGL14.eglCreateContext(
            display, configs[0], android.opengl.EGL14.EGL_NO_CONTEXT,
            ctxAttribs, 0)
        check(ctx != android.opengl.EGL14.EGL_NO_CONTEXT) {
            "eglCreateContext failed"
        }
        val surfAttribs = intArrayOf(
            android.opengl.EGL14.EGL_WIDTH, 1,
            android.opengl.EGL14.EGL_HEIGHT, 1,
            android.opengl.EGL14.EGL_NONE)
        val surf = android.opengl.EGL14.eglCreatePbufferSurface(
            display, configs[0], surfAttribs, 0)
        check(surf != android.opengl.EGL14.EGL_NO_SURFACE) {
            "eglCreatePbufferSurface failed"
        }
        check(android.opengl.EGL14.eglMakeCurrent(
            display, surf, surf, ctx)) { "eglMakeCurrent failed" }
        val tex = IntArray(1)
        android.opengl.GLES20.glGenTextures(1, tex, 0)
        check(tex[0] != 0) { "glGenTextures failed" }
        eglDisplay = display
        eglContext = ctx
        eglSurface = surf
        cameraTexId = tex[0]
    }

    private fun releaseEgl() {
        try {
            val d = eglDisplay ?: return
            android.opengl.EGL14.eglMakeCurrent(
                d, android.opengl.EGL14.EGL_NO_SURFACE,
                android.opengl.EGL14.EGL_NO_SURFACE,
                android.opengl.EGL14.EGL_NO_CONTEXT)
            eglSurface?.let {
                android.opengl.EGL14.eglDestroySurface(d, it)
            }
            eglContext?.let {
                android.opengl.EGL14.eglDestroyContext(d, it)
            }
            android.opengl.EGL14.eglTerminate(d)
        } catch (e: Exception) {
            /* tearing down regardless */
        }
        eglDisplay = null
        eglContext = null
        eglSurface = null
        cameraTexId = 0
    }

    fun start(onError: (String) -> Unit) {
        errorSink = onError
        running = true
        thread = Thread {
            try {
                runLoop()
            } catch (e: Exception) {
                note = "arcore failed: ${e.message}"
                if (bringUpLatch.count > 0) {
                    bringUpError = note
                    bringUpLatch.countDown()
                }
                errorSink?.invoke(note)
            }
        }.apply { start() }
    }

    private fun trackingName(state: TrackingState): String = when (state) {
        TrackingState.TRACKING -> "TRACKING"
        TrackingState.PAUSED -> "LIMITED"
        TrackingState.STOPPED -> "STOPPED"
    }

    private fun runLoop() {
        try {
            bringUp()
        } catch (e: Exception) {
            bringUpError = "arcore bring-up failed: ${e.message}"
            note = bringUpError!!
            bringUpLatch.countDown()
            releaseEgl()
            return
        }
        bringUpLatch.countDown()
        val sess = session!!
        val w = preparedW
        val h = preparedH
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, w, h)
        format.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
        )
        format.setInteger(MediaFormat.KEY_BIT_RATE, 10_000_000)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        val muxer = MediaMuxer(
            File(folder, "video.mp4").absolutePath,
            MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4
        )
        var track = -1
        var muxerStarted = false
        codec.start()
        val bufferInfo = MediaCodec.BufferInfo()
        var lastDepthNs = 0L
        var t0Ns: Long? = null
        encodedFrames = 0

        fun drain(endOfStream: Boolean) {
            if (endOfStream) {
                // Buffer-input codec: EOS is an empty buffer with
                // the flag, NOT signalEndOfInputStream() (that call
                // is surface-input only and throws here — in v0.3
                // it escaped the teardown and left a 0-byte video).
                try {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        codec.queueInputBuffer(
                            inIdx, 0, 0, 0,
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    }
                } catch (e: Exception) { /* codec already gone */ }
            }
            while (true) {
                val outIdx = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                when {
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        track = muxer.addTrack(codec.outputFormat)
                        muxer.start()
                        muxerStarted = true
                    }
                    outIdx >= 0 -> {
                        val buf = codec.getOutputBuffer(outIdx)!!
                        if (bufferInfo.size > 0 && muxerStarted) {
                            buf.position(bufferInfo.offset)
                            buf.limit(bufferInfo.offset + bufferInfo.size)
                            muxer.writeSampleData(track, buf, bufferInfo)
                            if (bufferInfo.flags and
                                MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0
                            ) {
                                encodedFrames++
                            }
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            return
                        }
                    }
                    else -> {
                        if (endOfStream) continue else return
                    }
                }
                if (!endOfStream) return
            }
        }

        var nUpdates = 0
        var nZeroTs = 0
        var nNotYet = 0
        var nNoInput = 0
        try {
            while (running) {
                val frame = try {
                    sess.update()
                } catch (e: CameraNotAvailableException) {
                    break
                }
                nUpdates++
                val ts = frame.timestamp
                if (ts == 0L) {
                    nZeroTs++
                    continue
                }
                if (t0Ns == null) t0Ns = ts
                if (intrinsics == null) {
                    try {
                        val intr = frame.camera.imageIntrinsics
                        intrinsics = JsonObj()
                            .put("source", "arcore-image-intrinsics")
                            .put("width", w)
                            .put("height", h)
                            .put("fx_px", intr.focalLength[0].toDouble())
                            .put("fy_px", intr.focalLength[1].toDouble())
                            .put("cx_px", intr.principalPoint[0].toDouble())
                            .put("cy_px", intr.principalPoint[1].toDouble())
                    } catch (e: Exception) {
                        /* intrinsics stay null; meta says so honestly */
                    }
                }
                // --- video frame -----------------------------------
                val image = try {
                    frame.acquireCameraImage()
                } catch (e: NotYetAvailableException) {
                    nNotYet++
                    continue
                }
                var queued = false
                val inIdx = codec.dequeueInputBuffer(10_000)
                if (inIdx < 0) nNoInput++
                if (inIdx >= 0) {
                    val inBuf = codec.getInputBuffer(inIdx)!!
                    val nv12 = yuv420ToNv12(image)
                    inBuf.clear()
                    inBuf.put(nv12, 0, minOf(nv12.size, inBuf.remaining()))
                    codec.queueInputBuffer(
                        inIdx, 0, minOf(nv12.size, inBuf.capacity()),
                        (ts - t0Ns!!) / 1000, 0
                    )
                    queued = true
                }
                if (queued && encodedFrames % 6 == 0) {
                    // Preview still from the very image just queued
                    // (before close) — the UI's live window on a
                    // camera ARCore will not share.
                    Thumbs.fromYuv(image, 256)?.let { onPreview(it) }
                }
                image.close()
                if (queued) {
                    onFrameStamp(ts)
                    // --- pose (aligned with the stamped frame) -----
                    onPose(ts, frame.camera.pose,
                        trackingName(frame.camera.trackingState))
                }
                drain(false)
                // --- depth ------------------------------------------
                if (depthActive && ts - lastDepthNs >= 100_000_000L) {
                    lastDepthNs = ts
                    try {
                        val dImg = frame.acquireDepthImage16Bits()
                        val plane = dImg.planes[0]
                        val shorts = ShortArray(plane.buffer.remaining() / 2)
                        plane.buffer.asShortBuffer().get(shorts)
                        if (plane.rowStride == dImg.width * 2) {
                            onDepth(ts, DepthDownsample.downsample(
                                shorts, dImg.width, dImg.height))
                        }
                        dImg.close()
                    } catch (e: NotYetAvailableException) {
                        /* no depth this frame yet */
                    }
                }
            }
        } finally {
            try { drain(true) } catch (e: Exception) { /* shutting down */ }
            try { codec.stop() } catch (e: Exception) { /* shutting down */ }
            codec.release()
            if (muxerStarted) muxer.stop()
            muxer.release()
            try { sess.pause() } catch (e: Exception) { /* shutting down */ }
            sess.close()
            releaseEgl()
        }
        note = "arcore session ended cleanly (depth=$depthActive, " +
            "encoded=$encodedFrames, updates=$nUpdates, " +
            "zero_ts=$nZeroTs, img_notyet=$nNotYet, " +
            "no_input=$nNoInput)"
    }

    /** Stops the session; returns the number of video samples the
     *  muxer actually wrote (the controller trims frames.csv to it). */
    fun stop(): Int {
        running = false
        thread?.join(5_000)
        thread = null
        return encodedFrames
    }

    /** YUV_420_888 -> NV12 byte array, honouring row/pixel strides. */
    private fun yuv420ToNv12(image: android.media.Image): ByteArray {
        val w = image.width
        val h = image.height
        val out = ByteArray(w * h * 3 / 2)
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        var pos = 0
        val yBuf = yPlane.buffer
        for (row in 0 until h) {
            for (col in 0 until w) {
                out[pos++] = yBuf.get(row * yPlane.rowStride + col * yPlane.pixelStride)
            }
        }
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        for (row in 0 until h / 2) {
            for (col in 0 until w / 2) {
                val uIdx = row * uPlane.rowStride + col * uPlane.pixelStride
                val vIdx = row * vPlane.rowStride + col * vPlane.pixelStride
                out[pos++] = uBuf.get(uIdx)
                out[pos++] = vBuf.get(vIdx)
            }
        }
        return out
    }
}
