package com.miaoxuan.mobile.model

import android.net.Uri

/** A user-selected image plus non-destructive, local-only culling metadata. */
data class PhotoCandidate(
    val uri: Uri,
    val displayName: String,
    val capturedAtMillis: Long,
    val groupId: Int = 0,
    val score: Double = 0.0,
    val recommended: Boolean = false,
    val sharpness: Double = 0.0,
    val luminance: Double = 0.0,
    val perceptualHash: Long = 0L,
    val reasons: List<String> = emptyList(),
)

data class PickerSettings(
    val minimumPerBurst: Int = 1,
    val maximumPerBurst: Int = 1,
    val targetKeepCount: Int? = null,
    val detectBlur: Boolean = true,
    val detectExposure: Boolean = true,
    val detectDuplicates: Boolean = true,
) {
    init {
        require(minimumPerBurst >= 1)
        require(maximumPerBurst >= minimumPerBurst)
        require(targetKeepCount == null || targetKeepCount > 0)
    }
}

data class TaskSummary(
    val title: String,
    val analyzedAtMillis: Long,
    val sourceCount: Int,
    val keepCount: Int,
)
