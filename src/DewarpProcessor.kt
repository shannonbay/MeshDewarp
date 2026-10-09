package com.github.shannonbay.rereader.util

import android.graphics.Bitmap
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.opencv.core.Core
import org.opencv.core.Point
import org.opencv.core.Size
import android.util.Log

class DewarpProcessor(
    private val dbNetModel: DBNetModel
) {

    companion object {
        private const val TAG = "DewarpProcessor"

        init {
            if (org.opencv.android.OpenCVLoader.initDebug()) {
                Log.i("OpenCV", "OpenCV loaded successfully")
            } else {
                Log.e("OpenCV", "OpenCV initialization failed!")
            }
        }
    }

    /**
     * Output mode for the dewarp pipeline.
     */
    enum class OutputMode {
        /** Returns the dewarped image */
        DEWARP,
        /** Returns the debug overlay visualization */
        OVERLAY,
        /** Returns a side-by-side comparison (original | dewarped) */
        COMPARISON
    }

    /**
     * Configuration for the dewarp pipeline.
     */
    data class DewarpConfig(
        val mode: OutputMode = OutputMode.DEWARP,
        val verticalSigma: Float = 200f,
        val xReliabilityScale: Float = 0.25f,
        val enableHorizontalCorrection: Boolean = false,
        val interpolation: Int = Imgproc.INTER_LINEAR,
        val debugLogs: Boolean = true
    )

    /**
     * Processes the input bitmap to dewarp text using detected baselines.
     *
     * @param original The warped input image
     * @param config Configuration options
     * @return The processed image (dewarped, overlay, or comparison based on mode)
     */
    fun process(original: Bitmap, config: DewarpConfig = DewarpConfig()): Bitmap {
        val totalStart = System.currentTimeMillis()
        val width = original.width
        val height = original.height

        Log.d(TAG, "========== Starting Dewarp Pipeline ==========")
        Log.d(TAG, "Input size: ${width}x${height}")
        Log.d(TAG, "Mode: ${config.mode}")

        // Stage 1: DBNet Heatmap Generation
        var stageStart = System.currentTimeMillis()
        val probMask = dbNetModel.getHeatmap(original)
        var stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[1/5] DBNet Heatmap: ${stageDuration}ms (size: ${probMask.width()}x${probMask.height()})")

        // Normalize coordinates - resize heatmap to match original dimensions
        stageStart = System.currentTimeMillis()
        if (probMask.width() != width || probMask.height() != height) {
            Log.d(TAG, "Resizing heatmap from ${probMask.width()}x${probMask.height()} to ${width}x${height}")
            Imgproc.resize(probMask, probMask, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        }
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[1b/5] Heatmap Resize: ${stageDuration}ms")

        // Stage 2: Thresholding to Binary
        stageStart = System.currentTimeMillis()
        val binary = Mat()
        Imgproc.threshold(probMask, binary, 0.3, 255.0, Imgproc.THRESH_BINARY)
        binary.convertTo(binary, CvType.CV_8UC1)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[2/5] Thresholding: ${stageDuration}ms")

        // Stage 2b: Morphological opening to break bridges between adjacent lines
        stageStart = System.currentTimeMillis()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, 3.0))
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[2b/5] Morphological Open: ${stageDuration}ms")

        val nonZeroBinary = Core.countNonZero(binary)
        val binaryPercent = (nonZeroBinary.toFloat() / binary.total()) * 100
        Log.d(TAG, "DIAGNOSTIC: Binary pixels: $nonZeroBinary (${String.format("%.2f", binaryPercent)}% of image)")

        if (nonZeroBinary == 0) {
            Log.e(TAG, "ERROR: Binary mask is completely empty! DBNet found no text or threshold too high.")
            probMask.release()
            binary.release()
            return original
        }

        // Stage 3: Extract Control Points via Center-of-Mass
        stageStart = System.currentTimeMillis()
        val linePoints = ExtractControlPointsUtil.extractControlPointsPipeline(binary)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[3/5] Control Point Extraction: ${stageDuration}ms (${linePoints.size} lines)")

        if (linePoints.isEmpty()) {
            Log.e(TAG, "No control points found! Check DBNet output or Thresholding.")
            Log.d(TAG, "========== Pipeline Aborted (No Data) ==========")
            probMask.release()
            binary.release()
            return original
        }

        // Stage 4: Splice line fragments
        stageStart = System.currentTimeMillis()
        val splicedLines = SplineSplicer.spliceLineFragments(linePoints)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[4/5] Line Splicing: ${stageDuration}ms (${splicedLines.size} spliced lines)")

        // Stage 5: Dewarp or Generate Overlay
        stageStart = System.currentTimeMillis()

        // Convert original Bitmap to Mat
        val sourceMat = Mat()
        org.opencv.android.Utils.bitmapToMat(original, sourceMat)

        // Create overlay params from config
        val overlayParams = TopologicalGridOverlay.OverlayParams(
            verticalSigma = config.verticalSigma,
            xReliabilityScale = config.xReliabilityScale,
            debugLogs = config.debugLogs
        )

        val overlay = TopologicalGridOverlay()

        val resultMat = when (config.mode) {
            OutputMode.DEWARP -> {
                overlay.dewarp(sourceMat, splicedLines, overlayParams, config.interpolation)
            }
            OutputMode.OVERLAY -> {
                overlay.generate(sourceMat, splicedLines, overlayParams)
            }
            OutputMode.COMPARISON -> {
                createComparisonImage(sourceMat, splicedLines, overlay, overlayParams, config)
            }
        }

        // Convert back to Bitmap
        val resultBitmap = Bitmap.createBitmap(resultMat.cols(), resultMat.rows(), Bitmap.Config.ARGB_8888)
        org.opencv.android.Utils.matToBitmap(resultMat, resultBitmap)

        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[5/5] ${config.mode}: ${stageDuration}ms")

        // Cleanup
        listOf(sourceMat, resultMat, probMask, binary).forEach { it.release() }

        Log.d(TAG, "========== Pipeline Complete: ${System.currentTimeMillis() - totalStart}ms ==========")
        return resultBitmap
    }

    /**
     * Legacy process method for backward compatibility.
     * Returns the debug overlay visualization.
     */
    fun process(original: Bitmap): Bitmap {
        return process(original, DewarpConfig(mode = OutputMode.OVERLAY))
    }

    /**
     * Convenience method to get the dewarped image directly.
     */
    fun dewarp(original: Bitmap): Bitmap {
        return process(original, DewarpConfig(mode = OutputMode.DEWARP))
    }

    /**
     * Creates a side-by-side comparison of original and dewarped images.
     */
    private fun createComparisonImage(
        sourceMat: Mat,
        baselines: List<List<Point>>,
        overlay: TopologicalGridOverlay,
        params: TopologicalGridOverlay.OverlayParams,
        config: DewarpConfig
    ): Mat {
        val width = sourceMat.cols()
        val height = sourceMat.rows()

        // Create dewarped version
        val dewarped = overlay.dewarp(sourceMat, baselines, params, config.interpolation)

        // Create combined image (side by side)
        val combined = Mat(height, width * 2, sourceMat.type())

        // Copy original to left half
        val leftRoi = combined.submat(0, height, 0, width)
        sourceMat.copyTo(leftRoi)

        // Copy dewarped to right half
        val rightRoi = combined.submat(0, height, width, width * 2)
        dewarped.copyTo(rightRoi)

        // Add labels
        Imgproc.putText(
            combined, "Original", Point(10.0, 30.0),
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0,
            org.opencv.core.Scalar(0.0, 0.0, 0.0), 3
        )
        Imgproc.putText(
            combined, "Original", Point(10.0, 30.0),
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0,
            org.opencv.core.Scalar(255.0, 255.0, 255.0), 2
        )

        Imgproc.putText(
            combined, "Dewarped", Point(width + 10.0, 30.0),
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0,
            org.opencv.core.Scalar(0.0, 0.0, 0.0), 3
        )
        Imgproc.putText(
            combined, "Dewarped", Point(width + 10.0, 30.0),
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0,
            org.opencv.core.Scalar(255.0, 255.0, 255.0), 2
        )

        // Draw separator line
        Imgproc.line(
            combined,
            Point(width.toDouble(), 0.0),
            Point(width.toDouble(), height.toDouble()),
            org.opencv.core.Scalar(128.0, 128.0, 128.0),
            2
        )

        dewarped.release()

        return combined
    }

    /**
     * Extracts control points by computing the vertical center-of-mass per column
     * directly from the binary mask.
     */
    @Deprecated("Use ExtractControlPointsUtil.extractControlPointsPipeline instead")
    fun extractControlPointsFromCenterOfMass(binary: Mat): List<List<Point>> {
        val start = System.currentTimeMillis()

        // Find connected components
        val labeled = Mat()
        val stats = Mat()
        val centroids = Mat()

        val ccStart = System.currentTimeMillis()
        val numLabels = Imgproc.connectedComponentsWithStats(binary, labeled, stats, centroids)
        val ccDuration = System.currentTimeMillis() - ccStart
        Log.v(TAG, "  Connected Components: $numLabels labels (${ccDuration}ms)")

        val rows = labeled.rows()
        val cols = labeled.cols()

        // Pre-allocate buffer for efficient pixel access
        val labelData = IntArray(rows * cols)
        labeled.get(0, 0, labelData)

        val allLines = mutableListOf<List<Point>>()

        // Filter valid labels - skip tiny components
        val minWidth = 50
        val minHeight = 3
        val maxAspectRatio = 50.0

        val validLabels = mutableListOf<Int>()
        for (label in 1 until numLabels) {
            val width = stats.get(label, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val height = stats.get(label, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val aspectRatio = width.toDouble() / height.coerceAtLeast(1)

            if (width >= minWidth && height >= minHeight && aspectRatio > 1.0 && aspectRatio < maxAspectRatio) {
                validLabels.add(label)
            }
        }

        Log.v(TAG, "  Processing ${validLabels.size} valid components (filtered from $numLabels)")

        for (label in validLabels) {
            val left = stats.get(label, Imgproc.CC_STAT_LEFT)[0].toInt()
            val top = stats.get(label, Imgproc.CC_STAT_TOP)[0].toInt()
            val width = stats.get(label, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val height = stats.get(label, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val right = left + width

            // Compute center-of-mass for each column within this component
            val columnCenters = mutableMapOf<Int, Double>()

            for (x in left until right.coerceAtMost(cols)) {
                var sumY = 0.0
                var count = 0

                for (y in top until (top + height).coerceAtMost(rows)) {
                    if (labelData[y * cols + x] == label) {
                        sumY += y
                        count++
                    }
                }

                if (count > 0) {
                    columnCenters[x] = sumY / count
                }
            }

            if (columnCenters.isEmpty()) continue

            // Sample control points at regular intervals
            val linePoints = mutableListOf<Point>()
            val step = (width / 10).coerceIn(10, 30)

            val sortedColumns = columnCenters.keys.sorted()
            if (sortedColumns.isEmpty()) continue

            // Add first point
            val firstX = sortedColumns.first()
            linePoints.add(Point(firstX.toDouble(), columnCenters[firstX]!!))

            // Add intermediate points at regular intervals
            var lastSampledX = firstX
            for (x in sortedColumns) {
                if (x - lastSampledX >= step) {
                    linePoints.add(Point(x.toDouble(), columnCenters[x]!!))
                    lastSampledX = x
                }
            }

            // Add last point if not already added
            val lastX = sortedColumns.last()
            if (lastX != lastSampledX) {
                linePoints.add(Point(lastX.toDouble(), columnCenters[lastX]!!))
            }

            // Only keep lines with enough points
            if (linePoints.size > 4) {
                allLines.add(linePoints)
            }
        }

        labeled.release()
        stats.release()
        centroids.release()

        val totalDuration = System.currentTimeMillis() - start
        Log.v(TAG, "  Center-of-mass extraction total: ${totalDuration}ms, ${allLines.size} lines")

        return allLines
    }
}