package com.github.shannonbay.rereader.util

import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * TopologicalGridOverlay constructs and visualizes a topology-preserving 2D grid
 * influenced by a set of horizontal baseline splines extracted from a warped document image.
 *
 * This class provides:
 * - Overlay visualization for debugging the deformation field
 * - Actual dewarping via remap matrices (optimized for large images)
 */
class TopologicalGridOverlay {

    companion object {
        private const val TAG = "TopologicalGridOverlay"

        // Colors (BGR format for OpenCV)
        private val COLOR_DISTORTED_GRID = Scalar(255.0, 0.0, 0.0)      // Blue
        private val COLOR_UNIFORM_GRID = Scalar(0.0, 0.0, 255.0)        // Red
        private val COLOR_BASELINE = Scalar(0.0, 255.0, 0.0)            // Green
        private val COLOR_INTERSECTION = Scalar(0.0, 255.0, 255.0)      // Yellow
        private val COLOR_MAPPING_ARROW = Scalar(255.0, 255.0, 0.0)     // Cyan
        private val COLOR_WARNING = Scalar(0.0, 0.0, 255.0)             // Red for warnings
        private val COLOR_TEXT = Scalar(255.0, 255.0, 255.0)            // White
    }

    /**
     * Configuration parameters for the overlay generation.
     */
    data class OverlayParams(
        val numCols: Int = 60,              // Number of vertical grid lines
        val numRows: Int = 100,             // Number of horizontal grid lines
        val samplesPerRow: Int = 40,        // Sampling resolution for horizontal curves
        val verticalSigma: Float = 200f,    // Vertical influence falloff (Gaussian sigma)
        val xReliabilityScale: Float = 0.25f, // Horizontal reliability decay factor
        val arcLengthSamples: Int = 100,    // Samples for arc length computation
        val drawIntersections: Boolean = true,
        val drawMappingArrows: Boolean = false,
        val debugLogs: Boolean = true
    )

    /**
     * Represents a fitted baseline spline with precomputed properties.
     */
    private data class FittedBaseline(
        val points: List<Point>,           // Original control points
        val xMin: Double,                  // Leftmost x coordinate
        val xMax: Double,                  // Rightmost x coordinate
        val yAtX: (Double) -> Double,      // Interpolation function y(x)
        val meanY: Double,                 // Mean y value (for vertical ordering)
        val yLUT: FloatArray? = null       // Precomputed Y lookup table (for fast access)
    )



    /**
     * Generates a topology overlay visualization.
     */
    fun generate(
        image: Mat,
        baselines: List<List<Point>>,
        params: OverlayParams = OverlayParams()
    ): Mat {
        val startTime = System.currentTimeMillis()

        if (params.debugLogs) {
            Log.d(TAG, "========== Generating Topology Overlay ==========")
            Log.d(TAG, "Image size: ${image.cols()}x${image.rows()}")
            Log.d(TAG, "Baselines: ${baselines.size}")
            Log.d(TAG, "Grid: ${params.numCols}x${params.numRows}")
        }

        val overlay = image.clone()

        if (baselines.isEmpty()) {
            if (params.debugLogs) {
                Log.w(TAG, "No baselines provided - returning original image")
            }
            drawText(overlay, "No baselines detected", Point(20.0, 40.0), COLOR_WARNING)
            return overlay
        }

        val fittedBaselines = fitBaselines(baselines, params)
        if (params.debugLogs) {
            Log.d(TAG, "Fitted ${fittedBaselines.size} valid baselines")
        }

        if (fittedBaselines.isEmpty()) {
            drawText(overlay, "Baseline fitting failed", Point(20.0, 40.0), COLOR_WARNING)
            return overlay
        }

        val width = image.cols()
        val height = image.rows()
        val uniformGrid = constructUniformGrid(width, height, params.numCols, params.numRows)

        val distortedGrid = constructDistortedGrid(
            uniformGrid, fittedBaselines, width, height, params
        )

        val topologyValid = verifyTopology(distortedGrid, params)
        val arcLengthStats = computeArcLengthStats(distortedGrid, params)

        renderOverlay(
            overlay, uniformGrid, distortedGrid, fittedBaselines,
            topologyValid, arcLengthStats, params
        )

        val duration = System.currentTimeMillis() - startTime
        if (params.debugLogs) {
            Log.d(TAG, "Overlay generation complete: ${duration}ms")
        }

        return overlay
    }

    // ========== OPTIMIZED DEWARPING ==========

    /**
     * Performs the actual dewarping of an image using the detected baselines.
     * Optimized for large images with many baselines.
     */
    fun dewarp(
        image: Mat,
        baselines: List<List<Point>>,
        params: OverlayParams = OverlayParams(),
        interpolation: Int = Imgproc.INTER_LINEAR
    ): Mat {
        val startTime = System.currentTimeMillis()
        val width = image.cols()
        val height = image.rows()

        if (params.debugLogs) {
            Log.d(TAG, "========== Starting Dewarp (Optimized) ==========")
            Log.d(TAG, "Image size: ${width}x${height}")
            Log.d(TAG, "Baselines: ${baselines.size}")
        }

        if (baselines.isEmpty()) {
            Log.w(TAG, "No baselines provided - returning original image")
            return image.clone()
        }

        // Generate the remap matrices using optimized method
        val (mapX, mapY) = generateRemapMatricesOptimized(width, height, baselines, params)

        val remapTime = System.currentTimeMillis()
        if (params.debugLogs) {
            Log.d(TAG, "Remap matrices generated: ${remapTime - startTime}ms")
        }

        // Perform the remap operation
        val dewarped = Mat()
        Imgproc.remap(
            image,
            dewarped,
            mapX,
            mapY,
            interpolation,
            org.opencv.core.Core.BORDER_REPLICATE
        )

        mapX.release()
        mapY.release()

        val duration = System.currentTimeMillis() - startTime
        if (params.debugLogs) {
            Log.d(TAG, "Dewarp complete: ${duration}ms (remap: ${System.currentTimeMillis() - remapTime}ms)")
        }

        return dewarped
    }

    /**
     * Optimized remap matrix generation using sparse grid + bilinear interpolation.
     *
     * Key optimizations:
     * 1. Build a sparse grid (numCols × numRows) with baseline-influenced Y offsets
     * 2. For each output pixel, find grid cell and bilinearly interpolate source position
     * 3. This reduces baseline iterations from O(width × height × baselines) to O(gridSize × baselines)
     *
     * For a 1848×4000 image with 30 baselines:
     * - Old: 7.4M × 30 = 222M baseline evaluations
     * - New: 6000 grid points × 30 = 180K baseline evaluations + 7.4M bilinear interpolations
     */
    fun generateRemapMatricesOptimized(
        width: Int,
        height: Int,
        baselines: List<List<Point>>,
        params: OverlayParams = OverlayParams()
    ): Pair<Mat, Mat> {
        val startTime = System.currentTimeMillis()

        val fittedBaselines = fitBaselines(baselines, params)
        val numBaselines = fittedBaselines.size

        if (numBaselines == 0) {
            return createIdentityRemapMatrices(width, height)
        }

        // Step 1: Build sparse distorted grid (source positions)
        val gridStart = System.currentTimeMillis()
        val numCols = params.numCols
        val numRows = params.numRows

        // Precompute Gaussian LUT
        val sigma = params.verticalSigma
        val sigma2x2 = 2.0f * sigma * sigma
        val maxDist = (3 * sigma).toInt().coerceAtMost(height)
        val gaussianLUT = FloatArray(maxDist + 1) { dist ->
            exp(-dist.toFloat() * dist.toFloat() / sigma2x2).toFloat()
        }

        // Build distorted grid: for each grid point, compute the source Y
        // distortedGrid[col][row] = Point(x, sourceY)
        val distortedGridY = Array(numCols) { FloatArray(numRows) }
        val uniformGridX = FloatArray(numCols) { col -> col * (width - 1).toFloat() / (numCols - 1) }
        val uniformGridY = FloatArray(numRows) { row -> row * (height - 1).toFloat() / (numRows - 1) }

        // Precompute baseline Y values at each grid column X
        val baselineYAtGridX = Array(numBaselines) { i ->
            FloatArray(numCols) { col ->
                fittedBaselines[i].yAtX(uniformGridX[col].toDouble()).toFloat()
            }
        }

        // Precompute baseline offsets (baselineY - meanY) at each grid column
        val baselineOffsetAtGridX = Array(numBaselines) { i ->
            val meanY = fittedBaselines[i].meanY.toFloat()
            FloatArray(numCols) { col -> baselineYAtGridX[i][col] - meanY }
        }

        // Precompute x-reliability at each grid column for each baseline
        val xReliabilityAtGridX = Array(numBaselines) { i ->
            val baseline = fittedBaselines[i]
            val xMin = baseline.xMin.toFloat()
            val xMax = baseline.xMax.toFloat()
            val baselineWidth = (xMax - xMin).coerceAtLeast(1f)
            val decayDistance = max(baselineWidth, params.xReliabilityScale * width)

            FloatArray(numCols) { col ->
                val x = uniformGridX[col]
                when {
                    x >= xMin && x <= xMax -> 1.0f
                    else -> {
                        val dist = if (x < xMin) xMin - x else x - xMax
                        1.0f / (1.0f + (dist / decayDistance) * (dist / decayDistance))
                    }
                }
            }
        }

        // Compute distorted grid Y values
        for (col in 0 until numCols) {
            for (row in 0 until numRows) {
                val dstY = uniformGridY[row]

                var totalWeight = 0.0f
                var weightedOffset = 0.0f

                for (i in 0 until numBaselines) {
                    val baselineY = baselineYAtGridX[i][col]
                    val vertDist = abs(dstY - baselineY).toInt()
                    val vertWeight = if (vertDist <= maxDist) gaussianLUT[vertDist] else 0f

                    if (vertWeight < 1e-6f) continue

                    val xReliability = xReliabilityAtGridX[i][col]
                    val weight = vertWeight * xReliability

                    if (weight > 1e-6f) {
                        val offset = baselineOffsetAtGridX[i][col]
                        totalWeight += weight
                        weightedOffset += weight * offset
                    }
                }

                val srcY = if (totalWeight > 1e-6f) {
                    dstY + weightedOffset / totalWeight
                } else {
                    dstY
                }

                distortedGridY[col][row] = srcY.coerceIn(0f, (height - 1).toFloat())
            }
        }

        if (params.debugLogs) {
            Log.d(TAG, "Build sparse grid (${numCols}×${numRows}): ${System.currentTimeMillis() - gridStart}ms")
        }

        // Step 2: Generate full remap matrices via bilinear interpolation
        val interpStart = System.currentTimeMillis()

        val mapX = Mat(height, width, CvType.CV_32FC1)
        val mapY = Mat(height, width, CvType.CV_32FC1)

        // Bulk arrays for efficient Mat access
        val xData = FloatArray(width * height)
        val yData = FloatArray(width * height)

        // Precompute grid cell boundaries for fast lookup
        val cellWidth = (width - 1).toFloat() / (numCols - 1)
        val cellHeight = (height - 1).toFloat() / (numRows - 1)
        val invCellWidth = 1.0f / cellWidth
        val invCellHeight = 1.0f / cellHeight

        for (dstY in 0 until height) {
            // Find row in grid
            val gridRowF = dstY * invCellHeight
            val rowIdx = gridRowF.toInt().coerceIn(0, numRows - 2)
            val ty = (gridRowF - rowIdx).coerceIn(0f, 1f)
            val ty1 = 1f - ty

            val rowOffset = dstY * width

            for (dstX in 0 until width) {
                // Find column in grid
                val gridColF = dstX * invCellWidth
                val colIdx = gridColF.toInt().coerceIn(0, numCols - 2)
                val tx = (gridColF - colIdx).coerceIn(0f, 1f)
                val tx1 = 1f - tx

                // Bilinear interpolation of source Y from grid
                val y00 = distortedGridY[colIdx][rowIdx]
                val y10 = distortedGridY[colIdx + 1][rowIdx]
                val y01 = distortedGridY[colIdx][rowIdx + 1]
                val y11 = distortedGridY[colIdx + 1][rowIdx + 1]

                val srcY = y00 * tx1 * ty1 + y10 * tx * ty1 + y01 * tx1 * ty + y11 * tx * ty

                val idx = rowOffset + dstX
                xData[idx] = dstX.toFloat()  // X unchanged (vertical-only correction)
                yData[idx] = srcY
            }
        }

        mapX.put(0, 0, xData)
        mapY.put(0, 0, yData)

        if (params.debugLogs) {
            Log.d(TAG, "Bilinear interpolation: ${System.currentTimeMillis() - interpStart}ms")
            Log.d(TAG, "Total remap generation: ${System.currentTimeMillis() - startTime}ms")
        }

        return Pair(mapX, mapY)
    }

    /**
     * Creates identity remap matrices (no transformation).
     */
    private fun createIdentityRemapMatrices(width: Int, height: Int): Pair<Mat, Mat> {
        val mapX = Mat(height, width, CvType.CV_32FC1)
        val mapY = Mat(height, width, CvType.CV_32FC1)

        val rowX = FloatArray(width) { it.toFloat() }
        val rowY = FloatArray(width)

        for (y in 0 until height) {
            for (x in 0 until width) {
                rowY[x] = y.toFloat()
            }
            mapX.put(y, 0, rowX)
            mapY.put(y, 0, rowY)
        }

        return Pair(mapX, mapY)
    }

    /**
     * Legacy remap generation method (slower, for reference).
     */
    fun generateRemapMatrices(
        width: Int,
        height: Int,
        baselines: List<List<Point>>,
        params: OverlayParams = OverlayParams()
    ): Pair<Mat, Mat> {
        // Delegate to optimized version
        return generateRemapMatricesOptimized(width, height, baselines, params)
    }

    // ========== BASELINE FITTING ==========

    /**
     * Fits baseline point lists to create interpolatable spline representations.
     * Uses binary search for faster interpolation.
     */
    private fun fitBaselines(
        baselines: List<List<Point>>,
        params: OverlayParams
    ): List<FittedBaseline> {
        return baselines.mapNotNull { points ->
            if (points.size < 2) {
                return@mapNotNull null
            }

            val sortedPoints = points.sortedBy { it.x }
            val xMin = sortedPoints.first().x
            val xMax = sortedPoints.last().x

            if (xMax - xMin < 50) {
                return@mapNotNull null
            }

            val meanY = sortedPoints.map { it.y }.average()

            // Precompute x-coordinates for binary search
            val xCoords = sortedPoints.map { it.x }.toDoubleArray()

            // Compute endpoint slopes for extrapolation
            val leftSlope = if (sortedPoints.size >= 3) {
                val p0 = sortedPoints[0]
                val p1 = sortedPoints[minOf(2, sortedPoints.size - 1)]
                if (p1.x != p0.x) (p1.y - p0.y) / (p1.x - p0.x) else 0.0
            } else 0.0

            val rightSlope = if (sortedPoints.size >= 3) {
                val pN = sortedPoints.last()
                val pN1 = sortedPoints[maxOf(0, sortedPoints.size - 3)]
                if (pN.x != pN1.x) (pN.y - pN1.y) / (pN.x - pN1.x) else 0.0
            } else 0.0

            // Create interpolation function using binary search
            val yAtX: (Double) -> Double = { x ->
                when {
                    x <= xMin -> {
                        sortedPoints.first().y + leftSlope * (x - xMin)
                    }
                    x >= xMax -> {
                        sortedPoints.last().y + rightSlope * (x - xMax)
                    }
                    else -> {
                        // Binary search for surrounding points
                        var lo = 0
                        var hi = xCoords.size - 1
                        while (lo < hi - 1) {
                            val mid = (lo + hi) / 2
                            if (xCoords[mid] <= x) lo = mid else hi = mid
                        }
                        val p0 = sortedPoints[lo]
                        val p1 = sortedPoints[hi]
                        val t = (x - p0.x) / (p1.x - p0.x)
                        p0.y + t * (p1.y - p0.y)
                    }
                }
            }

            FittedBaseline(
                points = sortedPoints,
                xMin = xMin,
                xMax = xMax,
                yAtX = yAtX,
                meanY = meanY
            )
        }.sortedBy { it.meanY }
    }

    // ========== GRID CONSTRUCTION ==========

    private fun constructUniformGrid(
        width: Int,
        height: Int,
        numCols: Int,
        numRows: Int
    ): Array<Array<Point>> {
        return Array(numRows) { row ->
            Array(numCols) { col ->
                Point(
                    col * (width - 1).toDouble() / (numCols - 1),
                    row * (height - 1).toDouble() / (numRows - 1)
                )
            }
        }
    }

    private fun constructDistortedGrid(
        uniformGrid: Array<Array<Point>>,
        baselines: List<FittedBaseline>,
        width: Int,
        height: Int,
        params: OverlayParams
    ): Array<Array<Point>> {
        val numRows = uniformGrid.size
        val numCols = uniformGrid[0].size
        val sigma2x2 = 2.0 * params.verticalSigma * params.verticalSigma

        return Array(numRows) { row ->
            Array(numCols) { col ->
                val uniformPoint = uniformGrid[row][col]
                val x = uniformPoint.x
                val uniformY = uniformPoint.y

                var totalWeight = 0.0
                var weightedOffset = 0.0

                for (baseline in baselines) {
                    val baselineY = baseline.yAtX(x)
                    val vertDist = abs(uniformY - baselineY)
                    val vertWeight = exp(-vertDist * vertDist / sigma2x2)

                    val xReliability = computeXReliability(x, baseline, params.xReliabilityScale, width)
                    val weight = vertWeight * xReliability

                    if (weight > 1e-6) {
                        val localOffset = baselineY - baseline.meanY
                        totalWeight += weight
                        weightedOffset += weight * localOffset
                    }
                }

                val finalY = if (totalWeight > 1e-6) {
                    uniformY + weightedOffset / totalWeight
                } else {
                    uniformY
                }

                Point(x, finalY.coerceIn(0.0, (height - 1).toDouble()))
            }
        }
    }

    private fun computeXReliability(
        x: Double,
        baseline: FittedBaseline,
        scale: Float,
        width: Int
    ): Double {
        if (x >= baseline.xMin && x <= baseline.xMax) {
            return 1.0
        }

        val dist = if (x < baseline.xMin) baseline.xMin - x else x - baseline.xMax
        val baselineWidth = baseline.xMax - baseline.xMin
        val decayDistance = max(baselineWidth, scale.toDouble() * width)
        return 1.0 / (1.0 + (dist / decayDistance) * (dist / decayDistance))
    }

    // ========== TOPOLOGY VERIFICATION ==========

    private fun verifyTopology(grid: Array<Array<Point>>, params: OverlayParams): Boolean {
        var inversions = 0
        val numRows = grid.size
        val numCols = grid[0].size

        for (col in 0 until numCols) {
            for (row in 1 until numRows) {
                if (grid[row][col].y <= grid[row - 1][col].y) {
                    inversions++
                }
            }
        }

        if (params.debugLogs && inversions > 0) {
            Log.w(TAG, "Topology violations: $inversions row inversions detected")
        }

        return inversions == 0
    }

    private fun computeArcLengthStats(
        grid: Array<Array<Point>>,
        params: OverlayParams
    ): ArcLengthStats {
        val numRows = grid.size
        val numCols = grid[0].size
        val arcLengths = DoubleArray(numRows)

        for (row in 0 until numRows) {
            var length = 0.0
            for (col in 1 until numCols) {
                val p0 = grid[row][col - 1]
                val p1 = grid[row][col]
                val dx = p1.x - p0.x
                val dy = p1.y - p0.y
                length += sqrt(dx * dx + dy * dy)
            }
            arcLengths[row] = length
        }

        val mean = arcLengths.average()
        val min = arcLengths.minOrNull() ?: 0.0
        val max = arcLengths.maxOrNull() ?: 0.0
        val stdDev = sqrt(arcLengths.map { (it - mean) * (it - mean) }.average())

        if (params.debugLogs) {
            Log.d(TAG, "Arc lengths - Mean: %.1f, Min: %.1f, Max: %.1f, StdDev: %.2f".format(
                mean, min, max, stdDev))
        }

        return ArcLengthStats(mean, min, max, stdDev, arcLengths.toList())
    }

    private data class ArcLengthStats(
        val mean: Double,
        val min: Double,
        val max: Double,
        val stdDev: Double,
        val perRow: List<Double>
    )

    // ========== RENDERING ==========

    private fun renderOverlay(
        overlay: Mat,
        uniformGrid: Array<Array<Point>>,
        distortedGrid: Array<Array<Point>>,
        baselines: List<FittedBaseline>,
        topologyValid: Boolean,
        arcLengthStats: ArcLengthStats,
        params: OverlayParams
    ) {
        val numRows = uniformGrid.size
        val numCols = uniformGrid[0].size

        drawGrid(overlay, uniformGrid, COLOR_UNIFORM_GRID, 1)
        drawGrid(overlay, distortedGrid, COLOR_DISTORTED_GRID, 2)

        for (baseline in baselines) {
            drawBaseline(overlay, baseline, COLOR_BASELINE, 3)
        }

        if (params.drawIntersections) {
            for (row in 0 until numRows step 5) {
                for (col in 0 until numCols step 5) {
                    val pt = distortedGrid[row][col]
                    Imgproc.circle(overlay, pt, 3, COLOR_INTERSECTION, -1)
                }
            }
        }

        if (params.drawMappingArrows) {
            for (row in 0 until numRows step 10) {
                for (col in 0 until numCols step 10) {
                    val src = uniformGrid[row][col]
                    val dst = distortedGrid[row][col]
                    Imgproc.arrowedLine(overlay, src, dst, COLOR_MAPPING_ARROW, 1)
                }
            }
        }

        val y = 30.0
        drawText(overlay, "Grid: ${numCols}x${numRows}", Point(10.0, y), COLOR_TEXT)
        drawText(overlay, "Baselines: ${baselines.size}", Point(10.0, y + 25), COLOR_TEXT)
        drawText(overlay, "Arc len: %.0f-%.0f (mean %.0f)".format(
            arcLengthStats.min, arcLengthStats.max, arcLengthStats.mean
        ), Point(10.0, y + 50), COLOR_TEXT)

        if (!topologyValid) {
            drawText(overlay, "WARNING: Topology violations!", Point(10.0, y + 75), COLOR_WARNING)
        }
    }

    private fun drawGrid(overlay: Mat, grid: Array<Array<Point>>, color: Scalar, thickness: Int) {
        val numRows = grid.size
        val numCols = grid[0].size

        for (row in 0 until numRows) {
            for (col in 1 until numCols) {
                Imgproc.line(overlay, grid[row][col - 1], grid[row][col], color, thickness)
            }
        }

        for (col in 0 until numCols) {
            for (row in 1 until numRows) {
                Imgproc.line(overlay, grid[row - 1][col], grid[row][col], color, thickness)
            }
        }
    }

    private fun drawBaseline(overlay: Mat, baseline: FittedBaseline, color: Scalar, thickness: Int) {
        val points = baseline.points
        for (i in 1 until points.size) {
            Imgproc.line(overlay, points[i - 1], points[i], color, thickness)
        }
    }

    private fun drawText(overlay: Mat, text: String, position: Point, color: Scalar) {
        Imgproc.putText(overlay, text, position, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(0.0, 0.0, 0.0), 3)
        Imgproc.putText(overlay, text, position, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    }
}