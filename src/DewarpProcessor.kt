package com.github.shannonbay.rereader.util

import android.graphics.Bitmap
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.opencv.core.Core
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import android.util.Log
import kotlin.math.pow
import kotlin.math.abs

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
     * Processes the input bitmap to flatten the page using YOLO segmentation.
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
        Log.d(TAG, "[1/8] DBNet Heatmap: ${stageDuration}ms (size: ${probMask.width()}x${probMask.height()})")

        // DIAGNOSTIC: Check heatmap values
        val heatmapStats = Core.minMaxLoc(probMask)
        Log.d(TAG, "DIAGNOSTIC: Heatmap range [${heatmapStats.minVal}, ${heatmapStats.maxVal}]")
        val nonZeroHeatmap = Core.countNonZero(probMask)
        Log.d(TAG, "DIAGNOSTIC: Non-zero heatmap pixels: $nonZeroHeatmap / ${probMask.total()}")

        // Normalize coordinates - resize heatmap to match original dimensions
        stageStart = System.currentTimeMillis()
        if (probMask.width() != width || probMask.height() != height) {
            Log.d(TAG, "Resizing heatmap from ${probMask.width()}x${probMask.height()} to ${width}x${height}")
            Imgproc.resize(probMask, probMask, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)  // Use INTER_LINEAR instead of default
        }
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[1b/8] Heatmap Resize: ${stageDuration}ms")

        // Stage 2: Thresholding to Binary
        stageStart = System.currentTimeMillis()
        val binary = Mat()
        Imgproc.threshold(probMask, binary, 0.3, 255.0, Imgproc.THRESH_BINARY)
        binary.convertTo(binary, CvType.CV_8UC1)
        stageDuration = System.currentTimeMillis() - stageStart

        // DIAGNOSTIC: Check binary mask
        val nonZeroBinary = Core.countNonZero(binary)
        val binaryPercent = (nonZeroBinary.toFloat() / binary.total()) * 100
        Log.d(TAG, "[2/8] Thresholding: ${stageDuration}ms")
        Log.d(TAG, "DIAGNOSTIC: Binary pixels: $nonZeroBinary (${String.format("%.2f", binaryPercent)}% of image)")

        if (nonZeroBinary == 0) {
            Log.e(TAG, "ERROR: Binary mask is completely empty! DBNet found no text or threshold too high.")
            probMask.release()
            binary.release()
            return original
        }

        // Stage 3-6: Extract Control Points via Center-of-Mass (replaces skeletonization)
        // This approach computes the vertical center-of-mass per column directly from
        // the binary mask, which is faster and produces smoother curves than morphological
        // skeletonization followed by gap bridging.
        stageStart = System.currentTimeMillis()
        val linePoints = extractControlPointsFromCenterOfMass(binary)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[3-6/8] Center-of-Mass Extraction: ${stageDuration}ms (${linePoints.size} lines)")

        // DIAGNOSTIC: Show sample line points
        if (linePoints.isNotEmpty()) {
            val firstLine = linePoints[0]
            Log.d(TAG, "DIAGNOSTIC: First line has ${firstLine.size} points")
            if (firstLine.size >= 2) {
                Log.d(TAG, "  Start: (${firstLine[0].x}, ${firstLine[0].y})")
                Log.d(TAG, "  End: (${firstLine.last().x}, ${firstLine.last().y})")
            }
        }

        if (linePoints.isEmpty()) {
            Log.e(TAG, "No control points found! Check DBNet output or Thresholding.")
            Log.d(TAG, "========== Pipeline Aborted (No Data) ==========")
            probMask.release()
            binary.release()
            return original
        }

        // Stage 7: Generate Dewarp Maps (Vector Field Approach)
        stageStart = System.currentTimeMillis()
        val (mapX, mapY) = generateDewarpMaps(linePoints, width, height)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[7/8] Field Map Generation: ${stageDuration}ms")

        // DIAGNOSTIC: Check map statistics
        val mapYStats = Core.minMaxLoc(mapY)
        val mapXStats = Core.minMaxLoc(mapX)
        Log.d(TAG, "DIAGNOSTIC: MapY range [${mapYStats.minVal}, ${mapYStats.maxVal}]")
        Log.d(TAG, "DIAGNOSTIC: MapX range [${mapXStats.minVal}, ${mapXStats.maxVal}]")

        // Check for identity mapping
        var isIdentity = true
        val sampleSize = 100
        for (i in 0 until sampleSize) {
            val row = (height * i) / sampleSize
            val col = (width * i) / sampleSize
            val expectedY = row.toDouble()
            val actualY = mapY.get(row, col)[0]
            if (abs(actualY - expectedY) > 0.1) {
                isIdentity = false
                break
            }
        }
        if (isIdentity) {
            Log.w(TAG, "WARNING: Dewarp map appears to be identity mapping! No correction will occur.")
        }

        // Stage 8: Remap (Apply the Warp)
        stageStart = System.currentTimeMillis()

        // Convert original Bitmap to Mat
        val sourceMat = Mat()
        org.opencv.android.Utils.bitmapToMat(original, sourceMat)
        val destinationMat = Mat(height, width, sourceMat.type())

        // Perform the actual dewarp
        Imgproc.remap(sourceMat, destinationMat, mapX, mapY, Imgproc.INTER_LINEAR)

        // Convert back to Bitmap
        val resultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        org.opencv.android.Utils.matToBitmap(destinationMat, resultBitmap)

        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[8/8] Remap & Convert: ${stageDuration}ms")

        // Cleanup
        listOf(sourceMat, destinationMat, mapX, mapY, probMask, binary).forEach { it.release() }

        Log.d(TAG, "========== Pipeline Complete: ${System.currentTimeMillis() - totalStart}ms ==========")
        return resultBitmap
    }

    fun generateDewarpMaps(allLines: List<List<Point>>, width: Int, height: Int): Pair<Mat, Mat> {
        val samples = buildSlopeSamples(allLines)

        if (samples.isEmpty()) {
            Log.w(TAG, "No slope samples - returning identity mapping")
            val mapX = Mat(height, width, CvType.CV_32FC1)
            val mapY = Mat(height, width, CvType.CV_32FC1)
            val xData = FloatArray(width * height)
            val yData = FloatArray(width * height)
            for (row in 0 until height) {
                for (col in 0 until width) {
                    val idx = row * width + col
                    xData[idx] = col.toFloat()
                    yData[idx] = row.toFloat()
                }
            }
            mapX.put(0, 0, xData)
            mapY.put(0, 0, yData)
            return mapX to mapY
        }

        // DIAGNOSTIC: Analyze slope distribution
        val slopes = samples.map { it.slope }
        val avgSlope = slopes.average()
        val maxSlope = slopes.maxOrNull() ?: 0f
        val minSlope = slopes.minOrNull() ?: 0f
        val slopesNearZero = slopes.count { abs(it) < 0.01f }

        Log.d(TAG, "DIAGNOSTIC: Slope statistics:")
        Log.d(TAG, "  Samples: ${samples.size}")
        Log.d(TAG, "  Avg slope: ${String.format("%.4f", avgSlope)}")
        Log.d(TAG, "  Range: [${String.format("%.4f", minSlope)}, ${String.format("%.4f", maxSlope)}]")
        Log.d(TAG, "  Near-zero slopes: $slopesNearZero / ${samples.size}")

        if (abs(avgSlope) < 0.001 && abs(maxSlope) < 0.01 && abs(minSlope) < 0.01) {
            Log.w(TAG, "WARNING: All detected slopes are near-zero! Text appears already straight.")
            Log.w(TAG, "  This will produce identity mapping. Check if image actually needs dewarping.")
        }

        Log.d(TAG, "Building slope grid with ${samples.size} samples")
        val slopeGrid = SlopeGrid(samples, width, height)

        val mapX = Mat(height, width, CvType.CV_32FC1)
        val mapY = Mat(height, width, CvType.CV_32FC1)

        val xData = FloatArray(width * height)
        val yData = FloatArray(width * height)

        // FIXED: Correct dewarp approach - for each destination pixel,
        // calculate the vertical offset based on the detected text line curvature
        var maxDisplacement = 0f
        var totalDisplacement = 0f
        var sampleCount = 0

        // Strength multiplier - 1.0x for exact correction, adjust if needed
        // Was 2.0x which was overcorrecting and inverting the curve
        val strengthMultiplier = 1.0f

        Log.d(TAG, "Applying vertical dewarp correction")
        Log.d(TAG, "Correction strength: ${strengthMultiplier}x")

        for (row in 0 until height) {
            for (col in 0 until width) {
                val idx = row * width + col

                // For each pixel, get the slope at this position
                val slope = slopeGrid.getInterpolatedSlope(col.toFloat(), row.toFloat())

                // Calculate vertical offset: how far from the current row should we pull pixels?
                // The slope tells us dy/dx, so over the width of the image, the total offset is:
                // offset = slope * (col - centerX) where centerX is the horizontal center
                val centerX = width / 2f
                val horizontalDistance = col.toFloat() - centerX
                val verticalOffset = slope * horizontalDistance * strengthMultiplier

                // Source Y is the current row adjusted by the vertical offset
                // INVERTED: Changed from minus to plus to dewarp in opposite direction
                val sourceY = (row.toFloat() + verticalOffset).coerceIn(0f, (height - 1).toFloat())

                xData[idx] = col.toFloat()
                yData[idx] = sourceY

                if (col % 100 == 0 && row % 100 == 0) {
                    val displacement = abs(verticalOffset)
                    maxDisplacement = maxOf(maxDisplacement, displacement)
                    totalDisplacement += displacement
                    sampleCount++
                }
            }
        }

        val avgDisplacement = if (sampleCount > 0) totalDisplacement / sampleCount else 0f
        Log.d(TAG, "DIAGNOSTIC: Y-displacement stats:")
        Log.d(TAG, "  Max displacement: ${String.format("%.2f", maxDisplacement)} pixels")
        Log.d(TAG, "  Avg displacement: ${String.format("%.2f", avgDisplacement)} pixels")

        if (maxDisplacement < 1.0f) {
            Log.w(TAG, "WARNING: Maximum displacement < 1 pixel. Dewarp will be imperceptible!")
        }

        mapX.put(0, 0, xData)
        mapY.put(0, 0, yData)
        return mapX to mapY
    }

    class SlopeGrid(val samples: List<SlopeSample>, val width: Int, val height: Int, val gridSize: Int = 20) {
        private val grid = FloatArray(gridSize * gridSize)
        private val cellW = width.toFloat() / gridSize
        private val cellH = height.toFloat() / gridSize

        init {
            for (gy in 0 until gridSize) {
                for (gx in 0 until gridSize) {
                    val centerX = (gx + 0.5f) * cellW
                    val centerY = (gy + 0.5f) * cellH
                    grid[gy * gridSize + gx] = calculateWeightedSlope(centerX, centerY)
                }
            }

            // DIAGNOSTIC: Check grid statistics
            val gridSlopes = grid.toList()
            val avgGridSlope = gridSlopes.average()
            val maxGridSlope = gridSlopes.maxOrNull() ?: 0f
            val minGridSlope = gridSlopes.minOrNull() ?: 0f
            Log.v(TAG, "  Grid slope range: [${String.format("%.4f", minGridSlope)}, ${String.format("%.4f", maxGridSlope)}], avg: ${String.format("%.4f", avgGridSlope)}")
        }

        private fun calculateWeightedSlope(tx: Float, ty: Float): Float {
            var totalWeight = 0f
            var weightedSlope = 0f

            // ADJUSTED: Smaller sigma for more localized influence
            // Was: height / 10 (~400 pixels) - too much smoothing
            // Now: height / 20 (~200 pixels) - more responsive to local curvature
            val sigmaSq = (height / 20f).pow(2)

            for (s in samples) {
                val distSq = (tx - s.x) * (tx - s.x) + (ty - s.y) * (ty - s.y)
                val weight = 1f / (distSq + sigmaSq)
                weightedSlope += s.slope * weight
                totalWeight += weight
            }
            return if (totalWeight > 0) weightedSlope / totalWeight else 0f
        }

        fun getInterpolatedSlope(x: Float, y: Float): Float {
            val gx = (x / cellW).coerceIn(0f, gridSize - 1.001f)
            val gy = (y / cellH).coerceIn(0f, gridSize - 1.001f)
            val ix = gx.toInt()
            val iy = gy.toInt()

            val fx = gx - ix
            val fy = gy - iy

            val v00 = grid[iy * gridSize + ix]
            val v10 = if (ix + 1 < gridSize) grid[iy * gridSize + (ix + 1)] else v00
            val v01 = if (iy + 1 < gridSize) grid[(iy + 1) * gridSize + ix] else v00
            val v11 = if (ix + 1 < gridSize && iy + 1 < gridSize)
                grid[(iy + 1) * gridSize + (ix + 1)] else v00

            val v0 = v00 * (1 - fx) + v10 * fx
            val v1 = v01 * (1 - fx) + v11 * fx

            return v0 * (1 - fy) + v1 * fy
        }
    }

    data class SlopeSample(val x: Float, val y: Float, val slope: Float)

    fun buildSlopeSamples(allLines: List<List<Point>>): List<SlopeSample> {
        val samples = mutableListOf<SlopeSample>()
        for (line in allLines) {
            for (i in 0 until line.size - 1) {
                val p1 = line[i]
                val p2 = line[i + 1]

                val dx = (p2.x - p1.x).toFloat()
                val dy = (p2.y - p1.y).toFloat()

                if (dx > 0) {
                    samples.add(SlopeSample(
                        x = ((p1.x + p2.x) / 2).toFloat(),
                        y = ((p1.y + p2.y) / 2).toFloat(),
                        slope = dy / dx
                    ))
                }
            }
        }
        Log.d(TAG, "Built ${samples.size} slope samples from ${allLines.size} lines")
        return samples
    }

    /**
     * Extracts control points by computing the vertical center-of-mass per column
     * directly from the binary mask. This replaces the expensive skeletonization
     * pipeline (skeletonize -> bridge gaps -> morphological close -> extract points).
     *
     * For each connected component in the binary mask:
     * 1. Find the bounding box
     * 2. For each column within the component, compute the vertical center-of-mass
     *    (weighted average of Y coordinates where pixels are white)
     * 3. Sample these center-of-mass values at regular intervals as control points
     *
     * This approach is faster and produces smoother curves since it naturally
     * averages across the thickness of text lines rather than trying to find
     * a single-pixel skeleton.
     */
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
        val minHeight = 3  // Need some vertical extent to compute center-of-mass
        val maxAspectRatio = 50.0  // Skip very thin vertical lines (likely not text)

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
            // Store as map of column -> centerY
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

            // Adaptive step based on component width (similar to original approach)
            val step = (width / 10).coerceIn(10, 30)

            // Get sorted columns for this component
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

    private fun findSkeletonYFast(
        columnMap: HashMap<Int, MutableList<Int>>,
        x: Int,
        previousY: Int = -1
    ): Int {
        // Search nearby columns
        for (dx in -3..3) {
            val currX = x + dx
            val ysAtColumn = columnMap[currX] ?: continue

            if (ysAtColumn.isEmpty()) continue

            if (previousY == -1) return ysAtColumn[0]

            // Find closest Y to previousY
            var bestY = ysAtColumn[0]
            var minDelta = Math.abs(bestY - previousY)

            for (y in ysAtColumn) {
                val delta = Math.abs(y - previousY)
                if (delta < minDelta) {
                    minDelta = delta
                    bestY = y
                }
            }
            return bestY
        }
        return -1
    }
}