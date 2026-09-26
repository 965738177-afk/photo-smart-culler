package com.miaoxuan.mobile.analysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PhotoMetricsTest {
    @Test fun `hamming distance counts changed hash bits`() {
        assertEquals(4, PhotoMetrics.hammingDistance(0b0011, 0b1100))
    }

    @Test fun `sharp well exposed photo outranks a blurry clipped photo`() {
        val strong = PhotoMetrics.qualityScore(230.0, 0.52, 0.01, 0.01, false, false)
        val weak = PhotoMetrics.qualityScore(18.0, 0.92, 0.38, 0.0, true, true)
        assertTrue(strong > weak)
    }

    @Test fun `similar frames inside burst interval are grouped`() {
        assertTrue(PhotoMetrics.shouldGroup(1_000, 8_000, 12))
        assertFalse(PhotoMetrics.shouldGroup(1_000, 10_000, 12))
        assertFalse(PhotoMetrics.shouldGroup(1_000, 8_000, 31))
    }
}
