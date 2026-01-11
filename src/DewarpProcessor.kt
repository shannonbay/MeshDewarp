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
     * Processes the input bitmap to visualize detected text baselines using TopologicalGridOverlay.
     */
    fun process(original: Bitmap): Bitmap {

        val totalStart = System.currentTimeMillis()
        val width = original.width
        val height = original.height

        Log.d(TAG, "========== Starting Dewarp Pipeline ==========")
        Log.d(TAG, "Input size: ${original.width}x${original.height}")

        // Stage 1: DBNet Heatmap Generation
        var stageStart = System.currentTimeMillis()
        val probMask = dbNetModel.getHeatmap(original)
        var stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[1/4] DBNet Heatmap: ${stageDuration}ms (size: ${probMask.width()}x${probMask.height()})")

        // Normalize coordinates - resize heatmap to match original dimensions
        stageStart = System.currentTimeMillis()
        if (probMask.width() != width || probMask.height() != height) {
            Log.d(TAG, "Resizing heatmap from ${probMask.width()}x${probMask.height()} to ${width}x${height}")
            Imgproc.resize(probMask, probMask, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        }
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[1b/4] Heatmap Resize: ${stageDuration}ms")

        // Stage 2: Thresholding to Binary
        stageStart = System.currentTimeMillis()
        val binary = Mat()
        Imgproc.threshold(probMask, binary, 0.3, 255.0, Imgproc.THRESH_BINARY)
        binary.convertTo(binary, CvType.CV_8UC1)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[2/4] Thresholding: ${stageDuration}ms")

        // Stage 2b: Morphological opening to break bridges between adjacent lines
        stageStart = System.currentTimeMillis()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(1.0, 3.0))
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[2b/4] Morphological Open: ${stageDuration}ms")

        val nonZeroBinary = Core.countNonZero(binary)
        val binaryPercent = (nonZeroBinary.toFloat() / binary.total()) * 100
        Log.d(TAG, "[2/4] Thresholding: ${stageDuration}ms")
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
        Log.d(TAG, "[3/4] Center-of-Mass Extraction: ${stageDuration}ms (${linePoints.size} lines)")

        if (linePoints.isEmpty()) {
            Log.e(TAG, "No control points found! Check DBNet output or Thresholding.")
            Log.d(TAG, "========== Pipeline Aborted (No Data) ==========")
            probMask.release()
            binary.release()
            return original
        }

        val splicedLines = SplineSplicer.spliceLineFragments(linePoints)

        // Stage 4: Generate TopologicalGridOverlay
        stageStart = System.currentTimeMillis()

        // Convert original Bitmap to Mat
        val sourceMat = Mat()
        org.opencv.android.Utils.bitmapToMat(original, sourceMat)

        // Generate overlay
        val overlay = TopologicalGridOverlay()
        val overlayMat = overlay.generate(sourceMat, splicedLines)

        // Convert back to Bitmap
        val resultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        org.opencv.android.Utils.matToBitmap(overlayMat, resultBitmap)

        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[4/4] TopologicalGridOverlay: ${stageDuration}ms")

        // Cleanup
        listOf(sourceMat, overlayMat, probMask, binary).forEach { it.release() }

        Log.d(TAG, "========== Pipeline Complete: ${System.currentTimeMillis() - totalStart}ms ==========")
        return resultBitmap
    }

    /**
     * Extracts control points by computing the vertical center-of-mass per column
     * directly from the binary mask.
     */
    @Deprecated("Use extractControlPointsPipeline instead")
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