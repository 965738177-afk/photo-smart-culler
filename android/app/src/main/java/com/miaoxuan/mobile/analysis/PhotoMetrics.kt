package com.miaoxuan.mobile.analysis

/** Pure math kept outside Android APIs so it can be regression-tested on the JVM. */
object PhotoMetrics {
    fun hammingDistance(left: Long, right: Long): Int = java.lang.Long.bitCount(left xor right)

    fun qualityScore(
        sharpness: Double,
        luminance: Double,
        highlightFraction: Double,
        shadowFraction: Double,
        blurDetected: Boolean,
        exposureDetected: Boolean,
    ): Double {
        val focusScore = (sharpness / 180.0).coerceIn(0.0, 1.0) * 58.0
        val brightnessScore = (1.0 - (luminance - 0.52).let { it * it * 3.2 })
            .coerceIn(0.0, 1.0) * 30.0
        val clippingPenalty = (highlightFraction + shadowFraction).coerceAtMost(1.0) * 28.0
        val technicalPenalty = (if (blurDetected) 18.0 else 0.0) + (if (exposureDetected) 12.0 else 0.0)
        return (12.0 + focusScore + brightnessScore - clippingPenalty - technicalPenalty).coerceIn(0.0, 100.0)
    }

    fun shouldGroup(
        previousTimestampMillis: Long,
        timestampMillis: Long,
        hashDistance: Int,
        burstGapSeconds: Int = 8,
        duplicateThreshold: Int = 18,
    ): Boolean {
        val nearInTime = timestampMillis - previousTimestampMillis <= burstGapSeconds * 1_000L
        return nearInTime && hashDistance <= duplicateThreshold
    }
}

