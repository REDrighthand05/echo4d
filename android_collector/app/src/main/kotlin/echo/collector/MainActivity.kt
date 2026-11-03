package echo.collector

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.graphics.drawable.GradientDrawable
import android.hardware.SensorManager
import android.media.ImageReader
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Size
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import java.io.File
import kotlin.math.max
import kotlin.math.sin
import kotlin.random.Random

/**
 * One screen, deliberately minimal: the capability line, one big
 * record button, elapsed time, per-stream counts, storage left, and
 * send-back. The recording itself lives in CaptureService (it must
 * survive the screen going off); this screen is only a view on it.
 * Capture settings are the capability probe's job, not the user's.
 *
 * The look is the product's one face (the Muse / ECHO glass
 * language shared with the desktop app): a light frosted sheet,
 * borderless floating components, and the blue graffiti reduced
 * to tiny matte dots behind the glass — no gloss, no chrome.
 */
class MainActivity : Activity() {

    private lateinit var capLine: TextView
    private lateinit var statusLine: TextView
    private lateinit var updateLine: TextView
    private lateinit var recordButton: Button
    private lateinit var sendButton: Button
    private lateinit var caps: Capabilities
    private lateinit var backdrop: BackdropView
    private lateinit var halo: View
    private lateinit var previewImage: ImageView
    private lateinit var previewHint: TextView
    private lateinit var viz: VizView
    private val uiHandler = Handler(Looper.getMainLooper())
    private var shownCapture: File? = null
    private var idleThread: HandlerThread? = null
    private var idleDevice: CameraDevice? = null
    private var idleSession: CameraCaptureSession? = null
    private var idleReader: ImageReader? = null
    @Volatile private var idleThumb: Bitmap? = null
    private var lastIdleThumbNs = 0L
    private val animTicker = object : Runnable {
        override fun run() {
            val rec = CaptureService.instance?.controller?.state ==
                CaptureController.State.RECORDING
            val t = System.currentTimeMillis() / 1000.0
            if (rec) {
                val s = 1.0 + 0.035 * sin(t * 2.6)
                halo.scaleX = s.toFloat()
                halo.scaleY = s.toFloat()
                halo.alpha = (0.35 + 0.25 * sin(t * 2.6)).toFloat()
            } else {
                halo.scaleX = 1f
                halo.scaleY = 1f
                halo.alpha = 0f
            }
            uiHandler.postDelayed(this, 50)
        }
    }
    private val ticker = object : Runnable {
        override fun run() {
            refreshStatus()
            uiHandler.postDelayed(this, 150)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        caps = Capabilities.probe(this)
        writeFirstRunManifest()

        val frame = FrameLayout(this).apply {
            setBackgroundColor(PAGE_BG)
        }
        backdrop = BackdropView(this)
        frame.addView(
            backdrop,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(56, 72, 56, 48)
        }
        frame.addView(root)
        capLine = TextView(this).apply {
            text = caps.summaryLine
            setTextColor(SUB)
            textSize = 14f
            background = glassCard()
            setPadding(36, 28, 36, 28)
        }
        root.addView(capLine)
        if (!Storage.sharedGranted()) {
            // One tap opens the system grant screen; with it,
            // captures land in the shared folder and Syncthing
            // carries them home automatically.
            capLine.text = caps.summaryLine +
                " · 点此开启自动同步"
            capLine.setOnClickListener {
                try {
                    startActivity(
                        Intent(
                            android.provider.Settings
                                .ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                            android.net.Uri.parse("package:$packageName")
                        )
                    )
                } catch (e: Exception) {
                    statusLine.text = "打开授权页失败： ${e.message}"
                }
            }
        }

        // ---- preview box: one thumbnail stream for both states
        // (idle: this activity's own guest reader; recording: the
        // backend's) -- a single display path, rotated upright in
        // software, no texture-transform guesswork ---------------
        val previewBox = FrameLayout(this).apply {
            background = glassCard()
            clipToOutline = true
        }
        previewImage = ImageView(this).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            visibility = View.GONE
        }
        previewBox.addView(previewImage, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT))
        previewHint = TextView(this).apply {
            text = "相机预览"
            setTextColor(SUB)
            textSize = 15f
            gravity = Gravity.CENTER
        }
        previewBox.addView(previewHint, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(
            previewBox,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 520
            ).apply { topMargin = 40 }
        )

        // ---- record button with a breathing halo ----------------
        val recordWrap = FrameLayout(this)
        halo = View(this).apply {
            background = pill(Color.argb(120, 0, 87, 217))
            alpha = 0f
        }
        recordWrap.addView(halo, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT))
        recordButton = Button(this).apply {
            text = "● 开始录制"
            textSize = 26f
            setTextColor(Color.WHITE)
            isAllCaps = false
            background = pill(BLUE)
            setOnClickListener { toggle() }
        }
        recordWrap.addView(recordButton, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT))
        root.addView(
            recordWrap,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 300
            ).apply { topMargin = 48; bottomMargin = 36 }
        )
        statusLine = TextView(this).apply {
            text = "就绪"
            setTextColor(INK)
            textSize = 17f
            gravity = Gravity.CENTER
            background = glassCard()
            setPadding(36, 36, 36, 36)
        }
        root.addView(statusLine)

        // update line: hidden unless the relay offers a newer
        // collector; tapping downloads, verifies and installs.
        updateLine = TextView(this).apply {
            setTextColor(BLUE)
            textSize = 14.5f
            gravity = Gravity.CENTER
            visibility = View.GONE
            setPadding(8, 14, 8, 6)
        }
        root.addView(updateLine)
        checkForUpdate()

        // ---- data visualisation: attitude + route trace ---------
        viz = VizView(this).apply { background = glassCard() }
        root.addView(
            viz,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 330
            ).apply { topMargin = 36 }
        )
        sendButton = Button(this).apply {
            text = "打包发送采集…"
            textSize = 18f
            setTextColor(INK)
            isAllCaps = false
            background = glassPill()
            setOnClickListener { pickCaptureToSend() }
        }
        root.addView(
            sendButton,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 160
            ).apply { topMargin = 40 }
        )
        setContentView(frame)

        val perms = arrayListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
        if (Build.VERSION.SDK_INT >= 33) {
            perms.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        requestPermissions(perms.toTypedArray(), 1)
        uiHandler.post(ticker)
        uiHandler.post(animTicker)
        noteRecoveredCaptures()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(
            requestCode, permissions, grantResults)
        val camIdx = permissions.indexOf(Manifest.permission.CAMERA)
        if (camIdx >= 0 && grantResults.getOrNull(camIdx) ==
            PackageManager.PERMISSION_GRANTED) {
            openIdlePreview()
        }
    }

    // ---- idle preview (Camera2 guest reader, framing only) -----
    // Same thumbnail path as recording: YUV stills, rotated
    // upright in software. (The TextureView transform this
    // replaced put the picture sideways on the user's phone.)

    @SuppressLint("MissingPermission") // gated by the toggle/perm flow
    private fun openIdlePreview() {
        if (idleDevice != null) return
        val svcState = CaptureService.instance?.controller?.state
        if (svcState == CaptureController.State.RECORDING) return
        if (checkSelfPermission(Manifest.permission.CAMERA) !=
            PackageManager.PERMISSION_GRANTED) return
        val cm = getSystemService(CameraManager::class.java) ?: return
        try {
            val ch = cm.getCameraCharacteristics(caps.chosenCameraId)
            val map = ch.get(
                android.hardware.camera2.CameraCharacteristics
                    .SCALER_STREAM_CONFIGURATION_MAP) ?: return
            val sizes = map.getOutputSizes(
                android.graphics.ImageFormat.YUV_420_888).toList()
            val size = sizes.filter { it.width <= 800 }
                .maxByOrNull { it.width * it.height }
                ?: sizes.minByOrNull { it.width * it.height } ?: return
            idleThread = HandlerThread("echo-preview").apply { start() }
            val handler = Handler(idleThread!!.looper)
            val reader = ImageReader.newInstance(
                size.width, size.height,
                android.graphics.ImageFormat.YUV_420_888, 3)
            idleReader = reader
            reader.setOnImageAvailableListener({ r ->
                val img = r.acquireLatestImage()
                    ?: return@setOnImageAvailableListener
                try {
                    val ts = img.timestamp
                    if (ts - lastIdleThumbNs >= 100_000_000L) {
                        lastIdleThumbNs = ts
                        Thumbs.fromYuv(img, 320)?.let { bmp ->
                            idleThumb = Thumbs.rotate(
                                bmp, caps.chosenSensorOrientation)
                        }
                    }
                } finally {
                    img.close()
                }
            }, handler)
            cm.openCamera(caps.chosenCameraId,
                object : CameraDevice.StateCallback() {
                    override fun onOpened(dev: CameraDevice) {
                        idleDevice = dev
                        try {
                            @Suppress("DEPRECATION")
                            dev.createCaptureSession(
                                listOf(reader.surface),
                                object :
                                    CameraCaptureSession.StateCallback() {
                                    override fun onConfigured(
                                        sess: CameraCaptureSession) {
                                        idleSession = sess
                                        val req = dev.createCaptureRequest(
                                            CameraDevice.TEMPLATE_PREVIEW)
                                        req.addTarget(reader.surface)
                                        req.set(
                                            CaptureRequest.CONTROL_AF_MODE,
                                            CaptureRequest
                                                .CONTROL_AF_MODE_CONTINUOUS_VIDEO)
                                        sess.setRepeatingRequest(
                                            req.build(), null, handler)
                                    }
                                    override fun onConfigureFailed(
                                        sess: CameraCaptureSession) {
                                        sess.close()
                                    }
                                }, handler)
                        } catch (e: Exception) {
                            /* preview is a guest; recording rules */
                        }
                    }
                    override fun onDisconnected(dev: CameraDevice) {
                        dev.close()
                        if (idleDevice == dev) idleDevice = null
                    }
                    override fun onError(dev: CameraDevice, error: Int) {
                        dev.close()
                        if (idleDevice == dev) idleDevice = null
                    }
                }, handler)
        } catch (e: Exception) {
            /* preview is a guest; recording rules */
        }
    }

    private fun closeIdlePreview() {
        try { idleSession?.close() } catch (e: Exception) { /* done */ }
        idleSession = null
        try { idleDevice?.close() } catch (e: Exception) { /* done */ }
        idleDevice = null
        try { idleReader?.close() } catch (e: Exception) { /* done */ }
        idleReader = null
        idleThumb = null
        idleThread?.quitSafely()
        idleThread = null
    }

    private fun writeFirstRunManifest() {
        try {
            val f = File(Storage.base(this), "capabilities.json")
            if (!f.exists()) f.writeText(caps.manifest.render())
        } catch (e: Exception) {
            /* the manifest also travels inside every capture's meta.json */
        }
    }

    private fun capturesDir(): File = Storage.capturesDir(this)

    // -- in-app update (relay first, CDN mirror second) -----------
    private fun checkForUpdate() {
        val current = try {
            @Suppress("DEPRECATION")
            packageManager.getPackageInfo(packageName, 0)
                .longVersionCode.toInt()
        } catch (e: Exception) {
            return
        }
        Thread {
            val offer = AppUpdate.fetchOffer(current) ?: return@Thread
            runOnUiThread {
                updateLine.text =
                    "采集器有新版 v${offer.version}，点这里更新" +
                        if (offer.notes.isNotEmpty()) {
                            "（${offer.notes}）"
                        } else ""
                updateLine.visibility = View.VISIBLE
                updateLine.setOnClickListener {
                    updateLine.isEnabled = false
                    statusLine.text = "正在下载新版…"
                    AppUpdate.download(
                        this, offer,
                        onProgress = { pct ->
                            runOnUiThread {
                                statusLine.text = "下载新版 $pct%"
                            }
                        },
                        onReady = { file, err ->
                            runOnUiThread {
                                if (file != null) {
                                    statusLine.text =
                                        "下载好了，跟着系统提示装上就行"
                                    AppUpdate.install(this, file)
                                } else {
                                    statusLine.text =
                                        err ?: "更新没下成"
                                    updateLine.isEnabled = true
                                }
                            }
                        }
                    )
                }
            }
        }.start()
    }

    private fun listCaptures(): List<File> =
        capturesDir().listFiles { f ->
            f.isDirectory && f.name.startsWith("capture_")
        }?.sortedByDescending { it.name } ?: emptyList()

    private fun isFinalized(folder: File): Boolean {
        val meta = File(folder, "meta.json")
        if (!meta.exists()) return false
        return try {
            meta.readText().contains("\"finalized\": true")
        } catch (e: Exception) {
            false
        }
    }

    /** Captures whose app died before stop() mark themselves
     *  finalized=false in meta.json (written at start). They are
     *  still readable on the desktop — say so, don't hide them. */
    private fun noteRecoveredCaptures() {
        val open = listCaptures().filter { !isFinalized(it) }
        if (open.isNotEmpty()) {
            statusLine.text = "发现 ${open.size} 个未封口的采集（数据已落盘，可直接发送）"
        }
    }

    private fun pickCaptureToSend() {
        val captures = listCaptures()
        if (captures.isEmpty()) {
            statusLine.text = "还没有采集"
            return
        }
        val labels = captures.map { f ->
            val mb = f.walkTopDown().filter { it.isFile }
                .sumOf { it.length() } / 1e6
            val tag = if (isFinalized(f)) "" else "（未封口）"
            "${f.name} · ${mb} MB$tag"
        }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("选择要发送的采集")
            .setItems(labels) { _, which ->
                try {
                    ShareExport.share(this, captures[which])
                } catch (e: Exception) {
                    statusLine.text = "打包失败： ${e.message}"
                }
            }
            .show()
    }

    private fun toggle() {
        val svc = CaptureService.instance
        val recording = svc?.controller?.state ==
            CaptureController.State.RECORDING
        if (!recording) {
            if (checkSelfPermission(Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED
            ) {
                statusLine.text = "需要相机权限"
                return
            }
            CaptureService.lastError = null
            CaptureService.pendingCaps = caps
            // The idle preview holds the camera; hand it over
            // before the recording backend asks for it.
            closeIdlePreview()
            recordButton.text = "■ 停止"
            uiHandler.postDelayed({
                val intent = Intent(this, CaptureService::class.java)
                    .setAction(CaptureService.ACTION_START)
                startForegroundService(intent)
            }, 350)
        } else {
            val intent = Intent(this, CaptureService::class.java)
                .setAction(CaptureService.ACTION_STOP)
            startService(intent)
        }
    }

    private fun refreshStatus() {
        val svc = CaptureService.instance
        val c = svc?.controller
        var hadError = false
        CaptureService.lastError?.let { err ->
            statusLine.text = "错误： $err"
            CaptureService.lastError = null
            hadError = true
        }
        if (c != null && c.state == CaptureController.State.RECORDING) {
            recordButton.text = "■ 停止"
            setRecordingStyle(true)
            backdrop.working = true
            val bmp = CaptureService.latestPreview
            if (bmp != null) {
                previewImage.setImageBitmap(bmp)
                previewImage.visibility = View.VISIBLE
                previewHint.visibility = View.GONE
            } else {
                previewImage.visibility = View.GONE
                previewHint.text = "相机启动中…"
                previewHint.visibility = View.VISIBLE
            }
            viz.update(c.orientation(), c.trail())
            val secs = c.elapsedMs / 1000
            val gb = c.storageLeftBytes() / 1e9
            statusLine.text =
                "%02d:%02d · 帧 %d · IMU %d · ToF %d · 位姿 %d · 剩余 %.1f GB".format(
                    secs / 60, secs % 60, c.frameCount, c.imuCount,
                    c.tofCount, c.poseCount, gb
                )
            return
        }
        recordButton.text = "● 开始录制"
        setRecordingStyle(false)
        backdrop.working = false
        val idleBmp = idleThumb
        if (idleBmp != null) {
            previewImage.setImageBitmap(idleBmp)
            previewImage.visibility = View.VISIBLE
            previewHint.visibility = View.GONE
        } else {
            previewImage.visibility = View.GONE
            previewHint.text = "相机预览"
            previewHint.visibility = View.VISIBLE
        }
        viz.update(null, emptyList())
        if (idleDevice == null) {
            openIdlePreview()
        }
        val last = CaptureService.lastCapture
        if (last != null && last != shownCapture) {
            shownCapture = last
            // An error (e.g. the no-frame watchdog) is the report
            // for this capture — don't paint over it with "saved".
            if (!hadError) {
                statusLine.text = "已保存： ${last.absolutePath}"
            }
        }
    }

    override fun onDestroy() {
        uiHandler.removeCallbacks(ticker)
        uiHandler.removeCallbacks(animTicker)
        closeIdlePreview()
        super.onDestroy()
    }

    // ---- the shared glass language (tokens match the desktop) --

    private fun glassCard(): GradientDrawable = GradientDrawable().apply {
        setColor(Color.argb(215, 255, 255, 255))
        cornerRadius = 56f
        setStroke(2, Color.argb(46, 20, 40, 90))
    }

    private fun glassPill(): GradientDrawable = GradientDrawable().apply {
        setColor(Color.argb(200, 255, 255, 255))
        cornerRadius = 999f
        setStroke(2, Color.argb(46, 20, 40, 90))
    }

    private fun pill(color: Int): GradientDrawable =
        GradientDrawable().apply {
            setColor(color)
            cornerRadius = 999f
        }

    private fun setRecordingStyle(recording: Boolean) {
        recordButton.background = pill(if (recording) INK else BLUE)
    }

    companion object {
        private val PAGE_BG = Color.rgb(233, 237, 243)
        private val INK = Color.rgb(20, 23, 26)
        private val SUB = Color.rgb(69, 79, 90)
        private val BLUE = Color.rgb(0, 87, 217)
    }
}

/**
 * The sheet's background, all in one view: the impressionist
 * backdrop (his own photo, painterly-rendered, shipped as an
 * asset), a frosted veil so the glass UI stays readable, the
 * die-cut sticker of the user himself bobbing gently at the
 * lower right, and the blue graffiti reduced to tiny matte
 * dots drifting slowly over everything. While a capture runs
 * the veil deepens (the desktop app's dark-glass working state,
 * phone-sized). Animation is a slow invalidate loop that lives
 * and dies with the view's attachment.
 */
private class BackdropView(context: Context) : View(context) {
    @Volatile var working = false
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val backdropBmp: Bitmap? = loadAsset("backdrop.jpg")
    private val stickerBmp: Bitmap? = loadAsset("sticker.png")
    private var attached = false
    private val loop = object : Runnable {
        override fun run() {
            if (!attached) return
            invalidate()
            // While a capture runs the sheet is nearly static;
            // redraw slowly and leave the UI thread to the
            // preview and the readouts.
            postDelayed(this, if (working) 400 else 100)
        }
    }

    private fun loadAsset(name: String): Bitmap? = try {
        context.assets.open(name).use { BitmapFactory.decodeStream(it) }
    } catch (e: Exception) {
        null
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        attached = true
        post(loop)
    }

    override fun onDetachedFromWindow() {
        attached = false
        removeCallbacks(loop)
        super.onDetachedFromWindow()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w == 0f || h == 0f) return
        val t = System.currentTimeMillis() / 1000.0
        // --- impressionist backdrop, centre-cropped ------------
        backdropBmp?.let { bmp ->
            val s = max(w / bmp.width, h / bmp.height)
            val dw = bmp.width * s
            val dh = bmp.height * s
            canvas.drawBitmap(bmp, (w - dw) / 2f, (h - dh) / 2f, paint)
        }
        // --- frosted veil ---------------------------------------
        paint.color = if (working) {
            Color.argb(168, 16, 20, 27)
        } else {
            Color.argb(112, 238, 242, 247)
        }
        paint.style = Paint.Style.FILL
        canvas.drawRect(0f, 0f, w, h, paint)
        // --- the sticker, bobbing at the lower right ------------
        stickerBmp?.let { bmp ->
            val sw = w * 0.44f
            val sh = bmp.height * (sw / bmp.width)
            val bob = (sin(t * 0.8) * 9).toFloat()
            canvas.drawBitmap(
                bmp, null,
                RectF(w - sw - w * 0.015f, h - sh + h * 0.012f + bob,
                    w - w * 0.015f, h + h * 0.012f + bob),
                paint)
        }
        // --- drifting matte dots (the graffiti, quiet) ----------
        val rnd = Random(20261103)
        var x = -20f
        while (x < w + 20f) {
            val drift = sin(t * 0.5 + x * 0.01).toFloat() * 7f
            val y = h * 0.15f +
                sin((x / w) * Math.PI * 1.15).toFloat() * h * 0.05f +
                rnd.nextFloat() * 26f + drift
            dot(canvas, x + rnd.nextFloat() * 14f, y,
                2.2f + rnd.nextFloat() * 3.6f,
                26 + rnd.nextInt(44))
            x += 24f + rnd.nextFloat() * 22f
        }
        x = -10f
        while (x < w + 10f) {
            val drift = sin(t * 0.4 + x * 0.008 + 2.0).toFloat() * 6f
            val y = h * 0.29f -
                sin((x / w) * Math.PI * 0.9 + 0.5).toFloat() * h * 0.04f +
                rnd.nextFloat() * 22f + drift
            dot(canvas, x + rnd.nextFloat() * 12f, y,
                2f + rnd.nextFloat() * 3f,
                18 + rnd.nextInt(32))
            x += 34f + rnd.nextFloat() * 30f
        }
    }

    private fun dot(canvas: Canvas, cx: Float, cy: Float, r: Float,
                    alpha: Int) {
        paint.color = Color.argb(alpha, 0, 87, 217)
        canvas.drawCircle(cx, cy, r, paint)
    }
}

/**
 * Live data visualisation, drawn on canvas: an attitude
 * indicator (pitch/roll from the IMU quaternion) on the left,
 * and the route trace (ARCore x/z, top-down) on the right.
 * Before poses exist the trace area says so honestly.
 */
private class VizView(context: Context) : View(context) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var pitch = 0f
    private var roll = 0f
    private var trailPts: List<DoubleArray> = emptyList()

    fun update(quat: DoubleArray?, trail: List<DoubleArray>) {
        if (quat != null && quat.size == 4) {
            val r = FloatArray(9)
            SensorManager.getRotationMatrixFromVector(
                r, floatArrayOf(
                    quat[1].toFloat(), quat[2].toFloat(),
                    quat[3].toFloat(), quat[0].toFloat()))
            val o = FloatArray(3)
            SensorManager.getOrientation(r, o)
            pitch = o[1]
            roll = o[2]
        } else {
            pitch = 0f
            roll = 0f
        }
        trailPts = trail
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w == 0f || h == 0f) return
        val ink = Color.rgb(20, 23, 26)
        val blue = Color.rgb(0, 87, 217)
        // ---- attitude ------------------------------------------
        val sq = h - 64f
        val cx = 32f + sq / 2f
        val cy = h / 2f
        val r = sq / 2f
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 3f
        paint.color = Color.argb(120, 20, 23, 26)
        canvas.drawCircle(cx, cy, r, paint)
        canvas.save()
        val clip = Path().apply { addCircle(cx, cy, r - 3f, Path.Direction.CW) }
        canvas.clipPath(clip)
        canvas.translate(cx, cy)
        canvas.rotate(-Math.toDegrees(roll.toDouble()).toFloat())
        val pitchPx = (pitch * 2.2f).coerceIn(-r * 0.75f, r * 0.75f)
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 5f
        paint.color = Color.argb(190, 20, 23, 26)
        canvas.drawLine(-r, pitchPx, r, pitchPx, paint)
        paint.strokeWidth = 3f
        paint.color = Color.argb(110, 20, 23, 26)
        canvas.drawLine(-r * 0.6f, pitchPx + r * 0.32f,
            r * 0.6f, pitchPx + r * 0.32f, paint)
        canvas.restore()
        // fixed symbol
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 6f
        paint.color = blue
        canvas.drawLine(cx - r * 0.42f, cy, cx - r * 0.12f, cy, paint)
        canvas.drawLine(cx + r * 0.12f, cy, cx + r * 0.42f, cy, paint)
        paint.style = Paint.Style.FILL
        canvas.drawCircle(cx, cy, 7f, paint)
        // ---- route trace ----------------------------------------
        val x0 = 32f + sq + 44f
        val x1 = w - 32f
        val y0 = 32f
        val y1 = h - 32f
        if (x1 - x0 < 60f) return
        if (trailPts.size < 2) {
            paint.style = Paint.Style.FILL
            paint.color = Color.argb(110, 69, 79, 90)
            paint.textSize = 30f
            paint.textAlign = Paint.Align.CENTER
            canvas.drawText("暂无路线", (x0 + x1) / 2f, h / 2f + 10f, paint)
            return
        }
        var minX = Double.MAX_VALUE
        var maxX = -Double.MAX_VALUE
        var minZ = Double.MAX_VALUE
        var maxZ = -Double.MAX_VALUE
        for (p in trailPts) {
            minX = minOf(minX, p[0]); maxX = maxOf(maxX, p[0])
            minZ = minOf(minZ, p[1]); maxZ = maxOf(maxZ, p[1])
        }
        val spanX = max(1e-6, maxX - minX)
        val spanZ = max(1e-6, maxZ - minZ)
        val scale = minOf(
            (x1 - x0 - 28f) / spanX, (y1 - y0 - 28f) / spanZ).toFloat()
        fun px(p: DoubleArray) = (x0 + 14f + (p[0] - minX) * scale).toFloat()
        fun py(p: DoubleArray) = (y1 - 14f - (p[1] - minZ) * scale).toFloat()
        val path = Path()
        trailPts.forEachIndexed { i, p ->
            if (i == 0) path.moveTo(px(p), py(p)) else path.lineTo(px(p), py(p))
        }
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 5f
        paint.strokeJoin = Paint.Join.ROUND
        paint.color = blue
        canvas.drawPath(path, paint)
        paint.style = Paint.Style.FILL
        paint.color = Color.argb(140, 69, 79, 90)
        canvas.drawCircle(px(trailPts.first()), py(trailPts.first()), 7f, paint)
        paint.color = blue
        val last = trailPts.last()
        canvas.drawCircle(px(last), py(last), 11f, paint)
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 4f
        paint.color = Color.WHITE
        canvas.drawCircle(px(last), py(last), 11f, paint)
        paint.color = ink
    }
}
