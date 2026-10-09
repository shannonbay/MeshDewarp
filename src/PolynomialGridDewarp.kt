package com.github.shannonbay.rereader.util

import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Alternative dewarp map generation using polynomial-based grid construction
 * influenced by detected text line splines.
 *
 * Unlike the slope-based approach, this creates an explicit 2D grid structure
 * where text lines influence the horizontal curvature across the image.
 * This gives topology-preserving transformation like the academic paper.
 */
class PolynomialGridDewarp {

    companion object {
        private const val TAG = "PolynomialGridDewarp"
    }

    data class CubicPolynomial(
        val a: Float,
        val b: Float,
        val c: Float,
        val d: Float,
        val minX: Float,  // Actual fitted range
        val maxX: Float,
        val centerX: Float // Center of the fitted segment
    ) {
        fun evaluate(x: Float): Float = a * x.pow(3) + b * x.pow(2) + c * x + d

        // Get derivative (slope) at x
        fun derivative(x: Float): Float = 3 * a * x.pow(2) + 2 * b * x + c

        // Check if x is within the fitted range
        fun isWithinFittedRange(x: Float): Boolean = x >= minX && x <= maxX

        /**
         * Get curvature offset at position x, even outside fitted range.
         * Uses the polynomial within range, and projects the trend outside.
         */
        fun getCurvatureOffset(x: Float, baselineY: Float): Float {
            if (isWithinFittedRange(x)) {
                // Within fitted range: use actual polynomial value
                return evaluate(x) - baselineY
            } else {
                // Outside fitted range: project using edge value + slope
                val edgeX = if (x < minX) minX else maxX
                val edgeY = evaluate(edgeX)
                val edgeSlope = derivative(edgeX)

                // Linear projection from edge
                val dx = x - edgeX
                val projectedY = edgeY + edgeSlope * dx

                // Dampen the projection with distance
                val distance = abs(dx)
                val dampingFactor = 1f / (1f + distance / 100f) // Decay over ~100 pixels

                return (projectedY - baselineY) * dampingFactor
            }
        }
    }

    /**
     * Generate dewarp maps using text-line-influenced grid construction.
     *
     * Strategy:
     * 1. Fit polynomials to detected text line segments
     * 2. Create regular horizontal grid lines across the image
     * 3. Each grid line's curvature is influenced by nearby text line polynomials
     * 4. Create perpendicular vertical grid lines
     * 5. Map distorted grid → uniform rectangular grid
     */
    fun generateDewarpMaps(
        allLines: List<List<Point>>,
        width: Int,
        height: Int
    ): Pair<Mat, Mat> {

        if (allLines.isEmpty()) {
            Log.w(TAG, "No text lines - returning identity mapping")
            return createIdentityMapping(width, height)
        }

        Log.d(TAG, "========== Polynomial Grid Dewarp ==========")
        Log.d(TAG, "Processing ${allLines.size} text line segments")

        // Step 1: Fit polynomials to all detected text line segments
        val textLinePolynomials = allLines.mapNotNull { line ->
            val avgY = line.map { it.y }.average().toFloat()
            val poly = fitCubicPolynomial(line)
            if (poly != null) {
                // Store Y position with polynomial
                Pair(avgY, poly)
            } else {
                null
            }
        }

        if (textLinePolynomials.isEmpty()) {
            Log.w(TAG, "Failed to fit any polynomials - returning identity mapping")
            return createIdentityMapping(width, height)
        }

        Log.d(TAG, "Fitted ${textLinePolynomials.size} text line polynomials")

        // Step 2: Create regular horizontal grid lines influenced by text lines
        val numHorizontalGridLines = 50  // Fixed number of grid lines
        val horizontalGridPolynomials = mutableListOf<CubicPolynomial>()

        for (i in 0 until numHorizontalGridLines) {
            val gridY = height * i.toFloat() / (numHorizontalGridLines - 1)

            // Create a grid line at this Y, influenced by nearby text lines
            val gridPoly = createInfluencedGridLine(gridY, textLinePolynomials, width, height)
            horizontalGridPolynomials.add(gridPoly)
        }

        Log.d(TAG, "Created ${horizontalGridPolynomials.size} horizontal grid lines")

        // Step 3: Create vertical grid lines (perpendicular to horizontal)
        val numVerticalGridLines = 30

        // Step 4: Build the SOURCE (distorted) grid
        // This represents where the curved text lines actually are in the INPUT image
        val distortedGrid = mutableListOf<List<Point>>()

        for (i in 0 until numVerticalGridLines) {
            val x = width * i.toFloat() / (numVerticalGridLines - 1)
            val verticalLine = mutableListOf<Point>()

            // For each horizontal grid line polynomial, find where it intersects this X
            for (poly in horizontalGridPolynomials) {
                val y = poly.evaluate(x).toDouble()
                verticalLine.add(Point(x.toDouble(), y))
            }

            distortedGrid.add(verticalLine)
        }

        Log.d(TAG, "Built SOURCE (distorted) grid: ${distortedGrid.size} vertical × ${distortedGrid[0].size} horizontal")

        // Log sample distorted grid positions
        if (distortedGrid.isNotEmpty() && distortedGrid[0].isNotEmpty()) {
            val midCol = distortedGrid.size / 2
            val firstRow = distortedGrid[midCol][0]
            val lastRow = distortedGrid[midCol][distortedGrid[midCol].size - 1]
            Log.d(TAG, "  Sample distorted column (center): Y ranges from ${firstRow.y.toInt()} to ${lastRow.y.toInt()}")
        }

        // Step 5: Create DESTINATION (uniform rectangular) target grid
        // This is where we want the text lines to be in the OUTPUT image (straight and evenly spaced)
        // The Y positions should match the AVERAGE Y of each detected text line group

        // Get the Y range covered by detected text
        val minDetectedY = textLinePolynomials.minOf { it.first }
        val maxDetectedY = textLinePolynomials.maxOf { it.first }

        Log.d(TAG, "Detected text Y range: [$minDetectedY, $maxDetectedY]")

        val uniformGrid = createUniformGrid(
            numVerticalGridLines,
            numHorizontalGridLines,
            0.0,
            width.toDouble(),
            minDetectedY.toDouble(),
            maxDetectedY.toDouble()
        )

        Log.d(TAG, "Built DESTINATION (uniform) grid: ${uniformGrid.size} vertical × ${uniformGrid[0].size} horizontal")

        // Log sample uniform grid positions
        if (uniformGrid.isNotEmpty() && uniformGrid[0].isNotEmpty()) {
            val midCol = uniformGrid.size / 2
            val firstRow = uniformGrid[midCol][0]
            val lastRow = uniformGrid[midCol][uniformGrid[midCol].size - 1]
            Log.d(TAG, "  Sample uniform column (center): Y ranges from ${firstRow.y.toInt()} to ${lastRow.y.toInt()}")
            Log.d(TAG, "  This will straighten the curved text into evenly-spaced horizontal lines")
        }

        // Step 6: Generate displacement maps
        // For each pixel in the OUTPUT (uniform), find which pixel in INPUT (distorted) to sample from

        // Step 6: Generate displacement maps from distorted → uniform grid
        val mapX = Mat(height, width, CvType.CV_32FC1)
        val mapY = Mat(height, width, CvType.CV_32FC1)

        val xData = FloatArray(width * height)
        val yData = FloatArray(width * height)

        var maxDisplacement = 0f
        var totalDisplacement = 0f
        var sampleCount = 0

        // For each pixel in output, find corresponding source pixel
        for (row in 0 until height) {
            for (col in 0 until width) {
                val idx = row * width + col

                val (sourceX, sourceY) = findSourcePixel(
                    col.toFloat(),
                    row.toFloat(),
                    uniformGrid,
                    distortedGrid
                )

                xData[idx] = sourceX.coerceIn(0f, (width - 1).toFloat())
                yData[idx] = sourceY.coerceIn(0f, (height - 1).toFloat())

                // Track displacement statistics
                if (col % 100 == 0 && row % 100 == 0) {
                    val displacement = sqrt(
                        ((sourceX - col) * (sourceX - col) + (sourceY - row) * (sourceY - row))
                    )
                    maxDisplacement = maxOf(maxDisplacement, displacement)
                    totalDisplacement += displacement
                    sampleCount++
                }
            }
        }

        val avgDisplacement = if (sampleCount > 0) totalDisplacement / sampleCount else 0f
        Log.d(TAG, "Displacement stats:")
        Log.d(TAG, "  Max: ${String.format("%.2f", maxDisplacement)} pixels")
        Log.d(TAG, "  Avg: ${String.format("%.2f", avgDisplacement)} pixels")

        if (maxDisplacement < 1.0f) {
            Log.w(TAG, "WARNING: Max displacement < 1 pixel - dewarp will be imperceptible!")
        }

        mapX.put(0, 0, xData)
        mapY.put(0, 0, yData)

        Log.d(TAG, "========== Polynomial Grid Complete ==========")
        return mapX to mapY
    }

    /**
     * Create a horizontal grid line at the specified Y position,
     * influenced by nearby text line polynomials.
     *
     * Strategy:
     * - Text lines can influence across the full width via projection
     * - Multiple segments at similar Y (e.g., left/right columns) blend their influence
     * - Closer text lines have stronger influence (Y-distance based)
     * - Segments outside their fitted range project their curvature with damping
     */
    private fun createInfluencedGridLine(
        targetY: Float,
        textLinePolynomials: List<Pair<Float, CubicPolynomial>>,
        width: Int,
        height: Int
    ): CubicPolynomial {

        // Sample points across the width, influenced by nearby text lines
        val numSamples = 20
        val samplePoints = mutableListOf<Point>()

        for (i in 0 until numSamples) {
            val x = width * i.toFloat() / (numSamples - 1)

            // Calculate Y at this X position, influenced by nearby text lines
            var weightedOffset = 0f
            var totalWeight = 0f

            // Influence radius: text lines within this distance affect the grid line
            val influenceRadius = height * 0.15f  // 15% of image height

            for ((textY, textPoly) in textLinePolynomials) {
                val yDistance = abs(targetY - textY)

                // Y-distance based weight: closer lines have more influence
                val yWeight = if (yDistance < influenceRadius) {
                    val normalized = yDistance / influenceRadius
                    (1f - normalized * normalized)  // Quadratic falloff
                } else {
                    0f
                }

                if (yWeight > 0) {
                    // Get curvature offset at this X position
                    // This handles both fitted range and projection outside it
                    val offset = textPoly.getCurvatureOffset(x, textY)

                    // Additional X-distance based weight for projection quality
                    // Segments closer to this X position have more reliable curvature
                    val xDistanceToSegment = if (textPoly.isWithinFittedRange(x)) {
                        0f // Within range = perfect confidence
                    } else {
                        // Distance from nearest edge of segment
                        if (x < textPoly.minX) {
                            textPoly.minX - x
                        } else {
                            x - textPoly.maxX
                        }
                    }

                    // X-distance weight: segments covering this X get full weight,
                    // projected segments lose weight with distance
                    val xWeight = 1f / (1f + xDistanceToSegment / 200f)

                    // Combined weight: both Y-proximity and X-reliability
                    val combinedWeight = yWeight * xWeight

                    weightedOffset += offset * combinedWeight
                    totalWeight += combinedWeight
                }
            }

            // Calculate final Y position
            val finalY = if (totalWeight > 0) {
                targetY + weightedOffset / totalWeight
            } else {
                targetY  // No influence = straight line
            }

            samplePoints.add(Point(x.toDouble(), finalY.toDouble()))
        }

        // Fit a polynomial to these influenced sample points
        // This polynomial spans the full width [0, width]
        val fittedPoly = fitCubicPolynomial(samplePoints)
        return if (fittedPoly != null) {
            // Update to span full width since this is a grid line
            CubicPolynomial(
                fittedPoly.a,
                fittedPoly.b,
                fittedPoly.c,
                fittedPoly.d,
                0f,
                width.toFloat(),
                width / 2f
            )
        } else {
            // Fallback: straight horizontal line
            CubicPolynomial(0f, 0f, 0f, targetY, 0f, width.toFloat(), width / 2f)
        }
    }

    /**
     * Fit a cubic polynomial to a line using least squares.
     */
    private fun fitCubicPolynomial(line: List<Point>): CubicPolynomial? {
        if (line.size < 4) return null

        val sortedLine = line.sortedBy { it.x }
        val n = sortedLine.size

        val minX = sortedLine.first().x.toFloat()
        val maxX = sortedLine.last().x.toFloat()
        val centerX = (minX + maxX) / 2f

        // Build least squares matrices for y = ax³ + bx² + cx + d
        val X = Array(n) { i ->
            val x = sortedLine[i].x.toFloat()
            floatArrayOf(x.pow(3), x.pow(2), x, 1f)
        }
        val y = FloatArray(n) { i -> sortedLine[i].y.toFloat() }

        return try {
            val coeffs = solveLeastSquares(X, y)
            CubicPolynomial(coeffs[0], coeffs[1], coeffs[2], coeffs[3], minX, maxX, centerX)
        } catch (e: Exception) {
            Log.v(TAG, "Failed to fit polynomial: ${e.message}")
            null
        }
    }

    /**
     * Solve least squares (X'X)β = X'y for β.
     */
    private fun solveLeastSquares(X: Array<FloatArray>, y: FloatArray): FloatArray {
        val n = X.size
        val m = X[0].size

        // Compute X'X
        val XtX = Array(m) { FloatArray(m) }
        for (i in 0 until m) {
            for (j in 0 until m) {
                var sum = 0f
                for (k in 0 until n) {
                    sum += X[k][i] * X[k][j]
                }
                XtX[i][j] = sum
            }
        }

        // Compute X'y
        val Xty = FloatArray(m)
        for (i in 0 until m) {
            var sum = 0f
            for (k in 0 until n) {
                sum += X[k][i] * y[k]
            }
            Xty[i] = sum
        }

        return gaussianElimination(XtX, Xty)
    }

    /**
     * Gaussian elimination for 4×4 system.
     */
    private fun gaussianElimination(A: Array<FloatArray>, b: FloatArray): FloatArray {
        val n = A.size
        val Ab = Array(n) { i -> A[i].copyOf(n + 1).also { it[n] = b[i] } }

        // Forward elimination
        for (i in 0 until n) {
            var maxRow = i
            for (k in i + 1 until n) {
                if (abs(Ab[k][i]) > abs(Ab[maxRow][i])) {
                    maxRow = k
                }
            }

            val temp = Ab[i]
            Ab[i] = Ab[maxRow]
            Ab[maxRow] = temp

            for (k in i + 1 until n) {
                val factor = Ab[k][i] / Ab[i][i]
                for (j in i until n + 1) {
                    Ab[k][j] -= factor * Ab[i][j]
                }
            }
        }

        // Back substitution
        val x = FloatArray(n)
        for (i in n - 1 downTo 0) {
            var sum = Ab[i][n]
            for (j in i + 1 until n) {
                sum -= Ab[i][j] * x[j]
            }
            x[i] = sum / Ab[i][i]
        }

        return x
    }

    /**
     * Create a uniform rectangular grid.
     */
    private fun createUniformGrid(
        numCols: Int,
        numRows: Int,
        minX: Double,
        maxX: Double,
        minY: Double,
        maxY: Double
    ): List<List<Point>> {
        val grid = mutableListOf<List<Point>>()

        for (col in 0 until numCols) {
            val x = minX + (maxX - minX) * col / (numCols - 1)
            val verticalLine = mutableListOf<Point>()

            for (row in 0 until numRows) {
                val y = minY + (maxY - minY) * row / (numRows - 1)
                verticalLine.add(Point(x, y))
            }

            grid.add(verticalLine)
        }

        return grid
    }

    /**
     * Find source pixel using bilinear interpolation within the grid.
     */
    private fun findSourcePixel(
        destX: Float,
        destY: Float,
        uniformGrid: List<List<Point>>,
        distortedGrid: List<List<Point>>
    ): Pair<Float, Float> {

        val numCols = uniformGrid.size
        val numRows = uniformGrid[0].size

        if (numCols < 2 || numRows < 2) {
            return Pair(destX, destY)
        }

        // Find grid cell containing this pixel
        var colIdx = 0
        for (i in 0 until numCols - 1) {
            val x0 = uniformGrid[i][0].x.toFloat()
            val x1 = uniformGrid[i + 1][0].x.toFloat()
            if (destX >= x0 && destX <= x1) {
                colIdx = i
                break
            }
        }

        var rowIdx = 0
        for (i in 0 until numRows - 1) {
            val y0 = uniformGrid[0][i].y.toFloat()
            val y1 = uniformGrid[0][i + 1].y.toFloat()
            if (destY >= y0 && destY <= y1) {
                rowIdx = i
                break
            }
        }

        colIdx = colIdx.coerceIn(0, numCols - 2)
        rowIdx = rowIdx.coerceIn(0, numRows - 2)

        // Get four corners of grid cell
        val u00 = uniformGrid[colIdx][rowIdx]
        val u10 = uniformGrid[colIdx + 1][rowIdx]
        val u01 = uniformGrid[colIdx][rowIdx + 1]

        val d00 = distortedGrid[colIdx][rowIdx]
        val d10 = distortedGrid[colIdx + 1][rowIdx]
        val d01 = distortedGrid[colIdx][rowIdx + 1]
        val d11 = distortedGrid[colIdx + 1][rowIdx + 1]

        // Interpolation weights
        val dx = (u10.x - u00.x).toFloat()
        val dy = (u01.y - u00.y).toFloat()

        val tx = if (abs(dx) > 0.001f) {
            ((destX - u00.x) / dx).toFloat().coerceIn(0f, 1f)
        } else {
            0f
        }

        val ty = if (abs(dy) > 0.001f) {
            ((destY - u00.y) / dy).toFloat().coerceIn(0f, 1f)
        } else {
            0f
        }

        // Bilinear interpolation
        val sourceX = (
                d00.x * (1 - tx) * (1 - ty) +
                        d10.x * tx * (1 - ty) +
                        d01.x * (1 - tx) * ty +
                        d11.x * tx * ty
                ).toFloat()

        val sourceY = (
                d00.y * (1 - tx) * (1 - ty) +
                        d10.y * tx * (1 - ty) +
                        d01.y * (1 - tx) * ty +
                        d11.y * tx * ty
                ).toFloat()

        return Pair(sourceX, sourceY)
    }

    private fun createIdentityMapping(width: Int, height: Int): Pair<Mat, Mat> {
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
}