package com.echo.collector

import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

/**
 * A CSV file writer built for high-frequency sensor callbacks.
 *
 * Design:
 * - Callers append into a pre-sized [StringBuilder] that lives inside this
 *   writer. The buffer is the only allocation that grows; it is created once
 *   (64 KB) and reused for the whole recording.
 * - [append] copies a [CharSequence] into the buffer under a short lock. It
 *   never touches the disk, so it is safe to call from a sensor callback,
 *   the ARCore update loop, or a Camera2 capture callback.
 * - [flush] is called by a background writer thread (about once a second).
 *   It drains the buffer to disk in one write and flushes the underlying
 *   [BufferedWriter]. No sample is ever lost: the buffer is only cleared
 *   after its contents have been handed to the writer.
 * - [close] does a final flush and closes the file. Call it exactly once,
 *   after all producers have stopped.
 *
 * Callers are expected to build each line in a reusable StringBuilder of
 * their own and pass it here, so no per-sample String is created.
 */
class CsvWriter(file: File, header: String) {

    private val lock = Any()
    private val buffer = StringBuilder(65536)
    private val writer = BufferedWriter(FileWriter(file), 32768)
    private var closed = false

    init {
        writer.write(header)
        writer.write('\n'.code)
    }

    /** Copy [text] into the in-memory buffer. Never blocks on I/O. */
    fun append(text: CharSequence) {
        synchronized(lock) {
            if (!closed) buffer.append(text)
        }
    }

    /** Drain the buffer to disk. Call from the writer thread only. */
    fun flush() {
        val chunk: String
        synchronized(lock) {
            if (buffer.isEmpty()) {
                writer.flush()
                return
            }
            chunk = buffer.toString()
            buffer.setLength(0)
        }
        writer.write(chunk)
        writer.flush()
    }

    /** Final flush + close. Call once, after all producers stopped. */
    fun close() {
        synchronized(lock) { closed = true }
        flush()
        writer.close()
    }
}
