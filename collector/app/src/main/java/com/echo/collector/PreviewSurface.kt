package com.echo.collector

import android.view.Surface

/**
 * Holds the live camera preview surface.
 *
 * The preview lives in MainActivity (a TextureView), but the camera is
 * owned by RecorderSession inside RecorderService. A Surface cannot be
 * sent through an Intent, so the Activity publishes it here and the
 * session picks it up when it builds the capture session. The surface is
 * cleared the moment the TextureView is destroyed, so the session never
 * holds a dead surface.
 */
object PreviewSurface {
    @Volatile
    var surface: Surface? = null
        private set

    @Volatile
    var width: Int = 0
        private set

    @Volatile
    var height: Int = 0
        private set

    fun publish(surface: Surface?, width: Int, height: Int) {
        this.surface = surface
        this.width = width
        this.height = height
    }

    fun clear() {
        surface = null
        width = 0
        height = 0
    }
}
