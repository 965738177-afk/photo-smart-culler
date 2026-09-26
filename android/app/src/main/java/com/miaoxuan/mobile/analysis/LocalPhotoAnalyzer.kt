package com.miaoxuan.mobile.analysis

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.OpenableColumns
import androidx.exifinterface.media.ExifInterface
import com.miaoxuan.mobile.model.PhotoCandidate
import com.miaoxuan.mobile.model.PickerSettings
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.math.max

/**
 * Fast, explainable first-pass analysis. It only decodes downscaled previews and never writes to
 * the source URI. Scores are suggestions, so every recommendation remains editable in review.
 */
class LocalPhotoAnalyzer(private val context: Context) {
    private val resolver: ContentResolver = context.contentResolver

    suspend fun analyze(
        sourceUris: List<Uri>,
        settings: PickerSettings,
        onProgress: (completed: Int, total: Int) -> Unit,
    ): List<PhotoCandidate> {
        val scanned = sourceUris.mapIndexed { index, uri ->
            currentCoroutineContext().ensureActive()
            val bitmap = decodePreview(uri)
            val metrics = calculateMetrics(bitmap)
            bitmap.recycle()
            onProgress(index + 1, sourceUris.size)
            PhotoCandidate(
                uri = uri,
                displayName = displayName(uri),
                capturedAtMillis = captureTime(uri, index),
                sharpness = metrics.sharpness,
                luminance = metrics.luminance,
                perceptualHash = metrics.hash,
                score = PhotoMetrics.qualityScore(
                    sharpness = metrics.sharpness,
                    luminance = metrics.luminance,
                    highlightFraction = metrics.highlightFraction,
                    shadowFraction = metrics.shadowFraction,
                    blurDetected = settings.detectBlur && metrics.sharpness < BLUR_FLOOR,
                    exposureDetected = settings.detectExposure && metrics.isSeverelyExposed,
                ),
                reasons = buildList {
                    if (settings.detectBlur && metrics.sharpness < BLUR_FLOOR) add("疑似虚焦或抖动")
                    if (settings.detectExposure && metrics.luminance > 0.74 && metrics.highlightFraction > 0.12) add("严重过曝")
                    if (settings.detectExposure && metrics.luminance < 0.22 && metrics.shadowFraction > 0.12) add("严重欠曝")
                },
            )
        }.sortedBy { it.capturedAtMillis }

        return rate(scanned, settings)
    }

    private fun rate(items: List<PhotoCandidate>, settings: PickerSettings): List<PhotoCandidate> {
        if (items.isEmpty()) return emptyList()
        val groups = mutableListOf<MutableList<PhotoCandidate>>()
        items.forEach { item ->
            val current = groups.lastOrNull()
            val previous = current?.lastOrNull()
            val joinsPrevious = previous != null && PhotoMetrics.shouldGroup(
                previousTimestampMillis = previous.capturedAtMillis,
                timestampMillis = item.capturedAtMillis,
                hashDistance = PhotoMetrics.hammingDistance(previous.perceptualHash, item.perceptualHash),
                duplicateThreshold = if (settings.detectDuplicates) DUPLICATE_THRESHOLD else 64,
            )
            if (current == null || !joinsPrevious) groups.add(mutableListOf())
            groups.last() += item
        }

        val suggested = groups.flatMapIndexed { groupIndex, group ->
            group.sortedByDescending { it.score }.mapIndexed { rank, item ->
                item.copy(
                    groupId = groupIndex + 1,
                    // Rank within a burst: a uniformly dark scene can still contain its best
                    // frame. Keep the technical warning visible for manual review.
                    recommended = rank < settings.maximumPerBurst && item.score >= 35.0,
                )
            }
        }.toMutableList()

        val target = settings.targetKeepCount ?: return suggested
        var overage = suggested.count { it.recommended } - target
        if (overage <= 0) return suggested

        // Preserve the leading shot from every scene before trimming weaker backups.
        val protectedUris = suggested.groupBy { it.groupId }
            .values.mapNotNull { group -> group.maxByOrNull { it.score }?.uri }.toSet()
        suggested.indices
            .filter { suggested[it].recommended && suggested[it].uri !in protectedUris }
            .sortedBy { suggested[it].score }
            .forEach { index ->
                if (overage-- > 0) {
                    suggested[index] = suggested[index].copy(recommended = false)
                }
            }
        return suggested
    }

    private fun decodePreview(uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        // In bounds-only mode BitmapFactory intentionally returns null because it does not
        // allocate a Bitmap. Treat a readable stream—not that return value—as success.
        val streamWasOpened = resolver.openInputStream(uri)?.use { input ->
            BitmapFactory.decodeStream(input, null, bounds)
            true
        } ?: false
        check(streamWasOpened) { "无法读取图片" }
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "不是可识别的图片" }
        var sample = 1
        while (max(bounds.outWidth / sample, bounds.outHeight / sample) > MAX_DIMENSION) sample *= 2
        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return resolver.openInputStream(uri)?.use { input ->
            BitmapFactory.decodeStream(input, null, options)
        } ?: error("无法解码图片")
    }

    private fun calculateMetrics(bitmap: Bitmap): ImageMetrics {
        val width = bitmap.width
        val height = bitmap.height
        val gray = IntArray(width * height)
        var total = 0L
        var highlights = 0
        var shadows = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = bitmap.getPixel(x, y)
                val value = ((pixel shr 16 and 0xff) * 299 + (pixel shr 8 and 0xff) * 587 + (pixel and 0xff) * 114) / 1000
                gray[y * width + x] = value
                total += value
                if (value >= 245) highlights++
                if (value <= 16) shadows++
            }
        }
        var squaredSum = 0.0
        var laplaceSum = 0.0
        var count = 0
        for (y in 1 until height - 1) for (x in 1 until width - 1) {
            val center = gray[y * width + x]
            val laplace = 4 * center - gray[y * width + x - 1] - gray[y * width + x + 1] -
                gray[(y - 1) * width + x] - gray[(y + 1) * width + x]
            laplaceSum += laplace
            squaredSum += laplace.toDouble() * laplace
            count++
        }
        val variance = if (count == 0) 0.0 else squaredSum / count - (laplaceSum / count).let { it * it }
        val pixels = width * height
        return ImageMetrics(
            sharpness = variance,
            luminance = total.toDouble() / pixels / 255.0,
            highlightFraction = highlights.toDouble() / pixels,
            shadowFraction = shadows.toDouble() / pixels,
            hash = perceptualHash(gray, width, height),
        )
    }

    private fun perceptualHash(gray: IntArray, width: Int, height: Int): Long {
        val samples = IntArray(64)
        var total = 0
        for (row in 0 until 8) for (column in 0 until 8) {
            val x = (column * (width - 1) / 7).coerceIn(0, width - 1)
            val y = (row * (height - 1) / 7).coerceIn(0, height - 1)
            val value = gray[y * width + x]
            samples[row * 8 + column] = value
            total += value
        }
        val average = total / samples.size
        return samples.foldIndexed(0L) { index, result, value ->
            if (value >= average) result or (1L shl index) else result
        }
    }

    private fun displayName(uri: Uri): String = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
        ?: uri.lastPathSegment?.substringAfterLast('/')
        ?: "未命名照片"

    private fun captureTime(uri: Uri, fallbackIndex: Int): Long {
        val exifMillis = runCatching {
            resolver.openFileDescriptor(uri, "r")?.use { descriptor ->
                ExifInterface(descriptor.fileDescriptor).getAttribute(ExifInterface.TAG_DATETIME_ORIGINAL)
            }
        }.getOrNull()?.let { timestamp ->
            SimpleDateFormat("yyyy:MM:dd HH:mm:ss", Locale.US).parse(timestamp)?.time
        }
        return exifMillis ?: System.currentTimeMillis() + fallbackIndex
    }

    private data class ImageMetrics(
        val sharpness: Double,
        val luminance: Double,
        val highlightFraction: Double,
        val shadowFraction: Double,
        val hash: Long,
    ) {
        val isSeverelyExposed: Boolean
            get() = (luminance > 0.74 && highlightFraction > 0.12) || (luminance < 0.22 && shadowFraction > 0.12)
    }

    private companion object {
        const val MAX_DIMENSION = 640
        const val BLUR_FLOOR = 55.0
        const val DUPLICATE_THRESHOLD = 18
    }
}
