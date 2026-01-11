package com.github.shannonbay.rereader.util

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * TopologicalGridOverlay constructs and visualizes a topology-preserving 2D grid
 * influenced by a set of horizontal baseline splines extracted from a warped document image.
 *
 * This class does NOT perform dewarping. Its sole responsibility is to compute the mapping
 * topology and render an overlay showing:
 * - The distorted grid implied by the baselines (blue)
 * - The uniform destination grid (red)
 * - The baseline splines (green)
 * - Optional diagnostic markers (yellow intersections, cyan arrows)
 *
 * The overlay is used to validate the correctness of the deformation field before
 * any remapping is attempted.
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
        val meanY: Double                  // Mean y value (for vertical ordering)
    )

    /**
     * Generates a topology overlay visualization.
     *
     * @param image The source image (used for dimensions and as background)
     * @param baselines List of baseline splines, each as a list of points
     * @param params Configuration parameters
     * @return Mat containing the overlay visualization
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

        // Create output overlay (clone of original image)
        val overlay = image.clone()

        // Handle empty baselines case
        if (baselines.isEmpty()) {
            if (params.debugLogs) {
                Log.w(TAG, "No baselines provided - returning original image")
            }
            drawText(overlay, "No baselines detected", Point(20.0, 40.0), COLOR_WARNING)
            return overlay
        }

        // Step 1: Fit baselines and create interpolation functions
        val fittedBaselines = fitBaselines(baselines, params)
        if (params.debugLogs) {
            Log.d(TAG, "Fitted ${fittedBaselines.size} valid baselines")
        }

        if (fittedBaselines.isEmpty()) {
            drawText(overlay, "Baseline fitting failed", Point(20.0, 40.0), COLOR_WARNING)
            return overlay
        }

        // Step 2: Construct uniform destination grid
        val width = image.cols()
        val height = image.rows()
        val uniformGrid = constructUniformGrid(width, height, params.numCols, params.numRows)

        // Step 3: Construct distorted grid influenced by baselines
        val distortedGrid = constructDistortedGrid(
            uniformGrid, fittedBaselines, width, height, params
        )

        // Step 4: Verify topology and collect diagnostics
        val topologyValid = verifyTopology(distortedGrid, params)

        // Step 5: Compute arc-length statistics
        val arcLengthStats = computeArcLengthStats(distortedGrid, params)

        // Step 6: Render the overlay
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

    /**
     * Fits baseline point lists to create interpolatable spline representations.
     */
    private fun fitBaselines(
        baselines: List<List<Point>>,
        params: OverlayParams
    ): List<FittedBaseline> {
        return baselines.mapNotNull { points ->
            if (points.size < 2) {
                if (params.debugLogs) {
                    Log.v(TAG, "Skipping baseline with ${points.size} points")
                }
                return@mapNotNull null
            }

            // Sort points by x coordinate
            val sortedPoints = points.sortedBy { it.x }

            val xMin = sortedPoints.first().x
            val xMax = sortedPoints.last().x

            // Skip if baseline is too short
            if (xMax - xMin < 50) {
                if (params.debugLogs) {
                    Log.v(TAG, "Skipping short baseline: width=${xMax - xMin}")
                }
                return@mapNotNull null
            }

            val meanY = sortedPoints.map { it.y }.average()

            // Compute endpoint slopes for extrapolation
            // Use a few points at each end to get a stable slope estimate
            val leftSlope = if (sortedPoints.size >= 3) {
                val p0 = sortedPoints[0]
                val p1 = sortedPoints[minOf(2, sortedPoints.size - 1)]
                if (p1.x != p0.x) (p1.y - p0.y) / (p1.x - p0.x) else 0.0
            } else {
                0.0
            }

            val rightSlope = if (sortedPoints.size >= 3) {
                val pN = sortedPoints.last()
                val pN1 = sortedPoints[maxOf(0, sortedPoints.size - 3)]
                if (pN.x != pN1.x) (pN.y - pN1.y) / (pN.x - pN1.x) else 0.0
            } else {
                0.0
            }

            // Create interpolation function with extrapolation beyond endpoints
            val yAtX: (Double) -> Double = { x ->
                when {
                    x <= xMin -> {
                        // Extrapolate using left endpoint slope
                        val leftY = sortedPoints.first().y
                        leftY + leftSlope * (x - xMin)
                    }
                    x >= xMax -> {
                        // Extrapolate using right endpoint slope
                        val rightY = sortedPoints.last().y
                        rightY + rightSlope * (x - xMax)
                    }
                    else -> {
                        // Find surrounding points and interpolate
                        val idx = sortedPoints.indexOfFirst { it.x >= x }
                        if (idx <= 0) {
                            sortedPoints.first().y
                        } else {
                            val p0 = sortedPoints[idx - 1]
                            val p1 = sortedPoints[idx]
                            val t = (x - p0.x) / (p1.x - p0.x)
                            p0.y + t * (p1.y - p0.y)
                        }
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
        }.sortedBy { it.meanY } // Sort by vertical position
    }

    /**
     * Constructs a uniform rectangular grid spanning the image.
     * Returns a 2D array of Points [row][col].
     */
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

    /**
     * Constructs a distorted grid by applying baseline influences to the uniform grid.
     * Each point's y-coordinate is modified based on nearby baseline curvature.
     */
    private fun constructDistortedGrid(
        uniformGrid: Array<Array<Point>>,
        baselines: List<FittedBaseline>,
        width: Int,
        height: Int,
        params: OverlayParams
    ): Array<Array<Point>> {
        val numRows = uniformGrid.size
        val numCols = uniformGrid[0].size

        return Array(numRows) { row ->
            Array(numCols) { col ->
                val uniformPoint = uniformGrid[row][col]
                val x = uniformPoint.x
                val uniformY = uniformPoint.y

                // Compute weighted influence from all baselines
                var totalWeight = 0.0
                var weightedOffset = 0.0

                for (baseline in baselines) {
                    // Compute vertical distance weight (Gaussian falloff)
                    val baselineY = baseline.yAtX(x)
                    val vertDist = abs(uniformY - baselineY)
                    val vertWeight = exp(-vertDist * vertDist / (2 * params.verticalSigma * params.verticalSigma))

                    // Compute horizontal reliability weight
                    // Points outside the baseline's x-range get reduced influence
                    val xReliability = computeXReliability(x, baseline, params.xReliabilityScale, width)

                    val weight = vertWeight * xReliability

                    if (weight > 1e-6) {
                        // The offset is the difference between the baseline's y at this x
                        // and where a straight line would be
                        val baselineMeanY = baseline.meanY
                        val localOffset = baselineY - baselineMeanY

                        totalWeight += weight
                        weightedOffset += weight * localOffset
                    }
                }

                // Apply weighted offset, with smooth fallback to identity
                val finalY = if (totalWeight > 1e-6) {
                    uniformY + weightedOffset / totalWeight
                } else {
                    uniformY
                }

                // Ensure y stays within image bounds
                Point(x, finalY.coerceIn(0.0, (height - 1).toDouble()))
            }
        }
    }

    /**
     * Computes horizontal reliability weight for a given x position relative to a baseline.
     *
     * The spline's influence is projected beyond its endpoints - we extrapolate the
     * curve and maintain full influence, but apply a gradual confidence decay for
     * points far outside the measured range.
     *
     * @param x Query x position
     * @param baseline The fitted baseline
     * @param scale Decay scale factor (fraction of image width)
     * @param width Image width
     * @return Weight in range [0, 1]
     */
    private fun computeXReliability(
        x: Double,
        baseline: FittedBaseline,
        scale: Float,
        width: Int
    ): Double {
        // Inside the baseline's x-range: full reliability
        if (x >= baseline.xMin && x <= baseline.xMax) {
            return 1.0
        }

        // Outside the range: gradual decay but still significant influence
        // This allows the spline curvature to project beyond its endpoints
        val dist = if (x < baseline.xMin) {
            baseline.xMin - x
        } else {
            x - baseline.xMax
        }

        // Use a slower decay to maintain influence projection
        // The decay distance is relative to the baseline's own width for adaptivity
        val baselineWidth = baseline.xMax - baseline.xMin
        val decayDistance = max(baselineWidth * 0.5, scale.toDouble() * width)

        // Sigmoid-like decay: maintains ~90% influence for nearby points,
        // then smoothly decays. This projects the spline influence outward.
        return 1.0 / (1.0 + (dist / decayDistance) * (dist / decayDistance))
    }

    /**
     * Verifies that the distorted grid maintains topological correctness:
     * - Monotonic vertical ordering (no row inversions)
     * - No horizontal curve crossings
     * Returns true if topology is valid, false if violations detected.
     */
    private fun verifyTopology(
        grid: Array<Array<Point>>,
        params: OverlayParams
    ): Boolean {
        var valid = true
        var inversions = 0

        val numRows = grid.size
        val numCols = grid[0].size

        // Check vertical monotonicity at each column
        for (col in 0 until numCols) {
            for (row in 1 until numRows) {
                if (grid[row][col].y <= grid[row - 1][col].y) {
                    inversions++
                    valid = false
                }
            }
        }

        if (params.debugLogs && inversions > 0) {
            Log.w(TAG, "Topology violations: $inversions row inversions detected")
        }

        return valid
    }

    /**
     * Computes arc-length statistics for each horizontal grid row.
     */
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

    /**
     * Renders all overlay elements onto the image.
     */
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

        // Draw uniform grid (red, thinner, behind distorted)
        drawGrid(overlay, uniformGrid, COLOR_UNIFORM_GRID, 1)

        // Draw distorted grid (blue, thicker)
        drawGrid(overlay, distortedGrid, COLOR_DISTORTED_GRID, 2)

        // Draw baseline splines (green)
        for (baseline in baselines) {
            drawBaseline(overlay, baseline, COLOR_BASELINE, 3)
        }

        // Draw intersections (yellow circles at distorted grid points)
        if (params.drawIntersections) {
            for (row in 0 until numRows step 5) { // Sample every 5th row for clarity
                for (col in 0 until numCols step 5) {
                    val pt = distortedGrid[row][col]
                    Imgproc.circle(overlay, pt, 3, COLOR_INTERSECTION, -1)
                }
            }
        }

        // Draw mapping arrows (cyan, from uniform to distorted)
        if (params.drawMappingArrows) {
            for (row in 0 until numRows step 10) {
                for (col in 0 until numCols step 10) {
                    val src = uniformGrid[row][col]
                    val dst = distortedGrid[row][col]
                    Imgproc.arrowedLine(overlay, src, dst, COLOR_MAPPING_ARROW, 1)
                }
            }
        }

        // Draw text annotations
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

    /**
     * Draws a grid on the overlay image.
     */
    private fun drawGrid(
        overlay: Mat,
        grid: Array<Array<Point>>,
        color: Scalar,
        thickness: Int
    ) {
        val numRows = grid.size
        val numCols = grid[0].size

        // Draw horizontal lines
        for (row in 0 until numRows) {
            for (col in 1 until numCols) {
                Imgproc.line(overlay, grid[row][col - 1], grid[row][col], color, thickness)
            }
        }

        // Draw vertical lines
        for (col in 0 until numCols) {
            for (row in 1 until numRows) {
                Imgproc.line(overlay, grid[row - 1][col], grid[row][col], color, thickness)
            }
        }
    }

    /**
     * Draws a baseline spline as a smooth curve.
     */
    private fun drawBaseline(
        overlay: Mat,
        baseline: FittedBaseline,
        color: Scalar,
        thickness: Int
    ) {
        val points = baseline.points
        for (i in 1 until points.size) {
            Imgproc.line(overlay, points[i - 1], points[i], color, thickness)
        }
    }

    /**
     * Draws text on the overlay with a background for readability.
     */
    private fun drawText(
        overlay: Mat,
        text: String,
        position: Point,
        color: Scalar
    ) {
        // Draw black background for readability
        Imgproc.putText(
            overlay, text, position,
            Imgproc.FONT_HERSHEY_SIMPLEX, 0.6,
            Scalar(0.0, 0.0, 0.0), 3
        )
        // Draw text
        Imgproc.putText(
            overlay, text, position,
            Imgproc.FONT_HERSHEY_SIMPLEX, 0.6,
            color, 1
        )
    }

    // ========== Public utility methods for future dewarping ==========

    /**
     * Generates the remap matrices (mapX, mapY) for actual dewarping.
     * This can be used with Imgproc.remap() to perform the dewarping.
     *
     * @return Pair of (mapX, mapY) matrices for use with Imgproc.remap()
     */
    fun generateRemapMatrices(
        width: Int,
        height: Int,
        baselines: List<List<Point>>,
        params: OverlayParams = OverlayParams()
    ): Pair<Mat, Mat> {
        val fittedBaselines = fitBaselines(baselines, params)

        val mapX = Mat(height, width, CvType.CV_32FC1)
        val mapY = Mat(height, width, CvType.CV_32FC1)

        // For each destination pixel, find the corresponding source pixel
        for (dstY in 0 until height) {
            for (dstX in 0 until width) {
                // Compute the inverse mapping: where in the distorted source
                // does this uniform destination pixel come from?

                var totalWeight = 0.0
                var weightedOffset = 0.0

                for (baseline in fittedBaselines) {
                    val baselineY = baseline.yAtX(dstX.toDouble())
                    val vertDist = abs(dstY - baselineY)
                    val vertWeight = exp(-vertDist * vertDist /
                            (2 * params.verticalSigma * params.verticalSigma))

                    val xReliability = computeXReliability(
                        dstX.toDouble(), baseline, params.xReliabilityScale, width)

                    val weight = vertWeight * xReliability

                    if (weight > 1e-6) {
                        val localOffset = baselineY - baseline.meanY
                        totalWeight += weight
                        weightedOffset += weight * localOffset
                    }
                }

                val srcY = if (totalWeight > 1e-6) {
                    dstY + weightedOffset / totalWeight
                } else {
                    dstY.toDouble()
                }

                mapX.put(dstY, dstX, floatArrayOf(dstX.toFloat()))
                mapY.put(dstY, dstX, floatArrayOf(srcY.toFloat()))
            }
        }

        return Pair(mapX, mapY)
    }
}