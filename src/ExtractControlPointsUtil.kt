package com.github.shannonbay.rereader.util

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

object ExtractControlPointsUtil {

    /**
     * Primary pipeline (skeleton-based): runs skeletonization, gap bridging,
     * morphological close, and control-point extraction.
     *
     * Returns the extracted lines as List<List<Point>>.
     * The internal skeleton Mat is cleaned up before returning.
     */
    fun extractControlPointsPipeline(binary: Mat, logTag: String = "ExtractControlPointsUtil"): List<List<Point>> {
        val startTotal = System.currentTimeMillis()

        // Stage 1: Skeletonization
        val skelStart = System.currentTimeMillis()
        val skeleton = skeletonize(binary)
        val skelDuration = System.currentTimeMillis() - skelStart
        Log.d(logTag, "[Stage 1] Skeletonization: ${skelDuration}ms")

        val nonZeroSkeleton = Core.countNonZero(skeleton)
        Log.d(logTag, "DIAGNOSTIC: Skeleton pixels: $nonZeroSkeleton")

        // Stage 2: Bridge Gaps (Optimized)
        val bridgeStart = System.currentTimeMillis()
        bridgeSkeletonGapsOptimized(skeleton)
        val bridgeDuration = System.currentTimeMillis() - bridgeStart
        Log.d(logTag, "[Stage 2] Bridge Gaps (Optimized): ${bridgeDuration}ms")

        // Stage 3: Morphological Close
        val morphStart = System.currentTimeMillis()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 1.0))
        Imgproc.morphologyEx(skeleton, skeleton, Imgproc.MORPH_CLOSE, kernel)
        val morphDuration = System.currentTimeMillis() - morphStart
        Log.d(logTag, "[Stage 3] Morphological Close: ${morphDuration}ms")

        // Stage 4: Extract Control Points (Optimized)
        val extractStart = System.currentTimeMillis()
        val linePoints = extractSplineControlPointsOptimized(skeleton)
        val extractDuration = System.currentTimeMillis() - extractStart
        Log.d(logTag, "[Stage 4] Extract Control Points: ${extractDuration}ms (${linePoints.size} lines)")

        val totalDuration = System.currentTimeMillis() - startTotal
        Log.d(logTag, "Stages 1-4 total: ${totalDuration}ms")

        // Clean up internal skeleton Mat before returning control points
        skeleton.release()

        return linePoints
    }

    /**
     * Alternative pipeline (center-of-mass): computes vertical center-of-mass per column
     * directly from the binary mask and returns extracted control points.
     *
     * This method does not return a skeleton Mat; it returns only the extracted lines.
     */
    @Deprecated("Use extractControlPointsPipeline instead")
    fun extractControlPointsFromCenterOfMass(binary: Mat, logTag: String = "ExtractControlPointsUtil"): List<List<Point>> {
        val start = System.currentTimeMillis()

        // Find connected components
        val labeled = Mat()
        val stats = Mat()
        val centroids = Mat()

        val ccStart = System.currentTimeMillis()
        val numLabels = Imgproc.connectedComponentsWithStats(binary, labeled, stats, centroids)
        val ccDuration = System.currentTimeMillis() - ccStart
        Log.v(logTag, "  Connected Components: $numLabels labels (${ccDuration}ms)")

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

        Log.v(logTag, "  Processing ${validLabels.size} valid components (filtered from $numLabels)")

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
        Log.v(logTag, "  Center-of-mass extraction total: ${totalDuration}ms, ${allLines.size} lines")

        return allLines
    }

    // --- Skeletonize (copied & unchanged logic) ---
    private fun skeletonize(input: Mat): Mat {
        val start = System.currentTimeMillis()
        val res = input.clone()
        val element = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, Size(3.0, 3.0))
        val skel = Mat.zeros(res.size(), res.type())
        val temp = Mat()
        val eroded = Mat()

        var iterations = 0
        var done = false
        while (!done) {
            Imgproc.erode(res, eroded, element)
            Imgproc.dilate(eroded, temp, element)
            Core.subtract(res, temp, temp)
            Core.bitwise_or(skel, temp, skel)
            eroded.copyTo(res)

            iterations++
            if (Core.countNonZero(res) == 0) done = true
        }

        val duration = System.currentTimeMillis() - start
        Log.v("ExtractControlPointsUtil", "  Skeletonize iterations: $iterations (${duration}ms)")

        temp.release()
        eroded.release()
        res.release()

        return skel
    }

    // --- Bridge gaps (optimized) ---
    private fun bridgeSkeletonGapsOptimized(skeleton: Mat, maxGap: Double = 5.0) {
        val start = System.currentTimeMillis()

        val downscaleFactor = 8.0
        val small = Mat()
        Imgproc.resize(skeleton, small, Size(), 1.0 / downscaleFactor, 1.0 / downscaleFactor, Imgproc.INTER_NEAREST)

        val labeled = Mat()
        val stats = Mat()
        val centroids = Mat()
        val numLabels = Imgproc.connectedComponentsWithStats(small, labeled, stats, centroids)

        val minComponentSize = 5
        val validLabels = mutableSetOf<Int>()
        for (label in 1 until numLabels) {
            val area = stats.get(label, Imgproc.CC_STAT_AREA)[0].toInt()
            if (area >= minComponentSize) {
                validLabels.add(label)
            }
        }

        val rows = labeled.rows()
        val cols = labeled.cols()

        val labelData = IntArray(rows * cols)
        labeled.get(0, 0, labelData)

        val cellSize = ((maxGap / downscaleFactor) * 2).toInt().coerceAtLeast(2)
        val gridCols = (cols + cellSize - 1) / cellSize
        val spatialGrid = HashMap<Int, MutableList<Point>>()

        fun getGridKey(x: Int, y: Int): Int {
            val gx = x / cellSize
            val gy = y / cellSize
            return gy * gridCols + gx
        }

        var terminalCount = 0

        for (label in validLabels) {
            val left = stats.get(label, Imgproc.CC_STAT_LEFT)[0].toInt()
            val top = stats.get(label, Imgproc.CC_STAT_TOP)[0].toInt()
            val width = stats.get(label, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val height = stats.get(label, Imgproc.CC_STAT_HEIGHT)[0].toInt()

            for (y in top until (top + height).coerceAtMost(rows)) {
                val rowStart = y * cols
                for (x in left until (left + width).coerceAtMost(cols)) {
                    if (labelData[rowStart + x] != label) continue

                    var neighborCount = 0
                    for (dy in -1..1) {
                        val ny = y + dy
                        if (ny < 0 || ny >= rows) continue
                        val neighborRowStart = ny * cols

                        for (dx in -1..1) {
                            if (dx == 0 && dy == 0) continue
                            val nx = x + dx
                            if (nx >= 0 && nx < cols) {
                                if (labelData[neighborRowStart + nx] == label) {
                                    neighborCount++
                                }
                            }
                        }
                    }

                    if (neighborCount == 1) {
                        val fullResPoint = Point(x * downscaleFactor, y * downscaleFactor)
                        val key = getGridKey(x, y)
                        spatialGrid.getOrPut(key) { mutableListOf() }.add(fullResPoint)
                        terminalCount++
                    }
                }
            }
        }

        var bridgeCount = 0
        val maxGapSq = maxGap * maxGap

        spatialGrid.forEach { (key, points) ->
            val gy = key / gridCols
            val gx = key % gridCols

            for (dy in -1..1) {
                for (dx in -1..1) {
                    val neighborKey = (gy + dy) * gridCols + (gx + dx)
                    val neighborPoints = spatialGrid[neighborKey] ?: continue

                    val checkSameCell = (dy == 0 && dx == 0)

                    for (i in points.indices) {
                        val p1 = points[i]
                        val startJ = if (checkSameCell) i + 1 else 0

                        for (j in startJ until neighborPoints.size) {
                            val p2 = neighborPoints[j]

                            val dx2 = p1.x - p2.x
                            val dy2 = p1.y - p2.y
                            val distSq = dx2 * dx2 + dy2 * dy2

                            if (distSq > 0 && distSq <= maxGapSq) {
                                Imgproc.line(skeleton, p1, p2, Scalar(255.0), 1)
                                bridgeCount++
                            }
                        }
                    }
                }
            }
        }

        small.release()
        labeled.release()
        stats.release()
        centroids.release()

        val totalDuration = System.currentTimeMillis() - start
        Log.v("ExtractControlPointsUtil", "  Bridge total: ${totalDuration}ms, bridges: $bridgeCount, terminals: $terminalCount")
    }

    // --- Extract control points (optimized) ---
    // --- Extract control points (optimized) ---
    private fun extractSplineControlPointsOptimized(skeleton: Mat): List<List<Point>> {
        val start = System.currentTimeMillis()
        val labeled = Mat()
        val stats = Mat()
        val centroids = Mat()

        val ccStart = System.currentTimeMillis()
        val numLabels = Imgproc.connectedComponentsWithStats(skeleton, labeled, stats, centroids)
        val ccDuration = System.currentTimeMillis() - ccStart
        Log.v("ExtractControlPointsUtil", "  Connected Components: $numLabels labels (${ccDuration}ms)")

        val allLines = mutableListOf<List<Point>>()
        val extractStart = System.currentTimeMillis()

        val rows = labeled.rows()
        val cols = labeled.cols()

        val minWidth = 50
        val validLabels = mutableListOf<Int>()
        for (i in 1 until numLabels) {
            val width = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            if (width >= minWidth) {
                validLabels.add(i)
            }
        }

        val labelData = IntArray(rows * cols)
        labeled.get(0, 0, labelData)

        for (i in validLabels) {
            val linePoints = mutableListOf<Point>()
            val left = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val top = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val width = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val height = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val right = left + width - 1

            val step = (width / 10).coerceIn(10, 30)
            var previousY = -1

            val columnMap = HashMap<Int, MutableList<Int>>()
            for (y in top until (top + height).coerceAtMost(rows)) {
                val rowStart = y * cols
                for (x in left until (left + width).coerceAtMost(cols)) {
                    if (labelData[rowStart + x] == i) {
                        columnMap.getOrPut(x) { mutableListOf() }.add(y)
                    }
                }
            }

            val firstY = findSkeletonYFast(columnMap, left, previousY)
            if (firstY != -1) {
                linePoints.add(Point(left.toDouble(), firstY.toDouble()))
                previousY = firstY
            }

            for (x in (left + step) until (right - step) step step) {
                val y = findSkeletonYFast(columnMap, x, previousY)
                if (y != -1) {
                    linePoints.add(Point(x.toDouble(), y.toDouble()))
                    previousY = y
                }
            }

            val lastY = findSkeletonYFast(columnMap, right, previousY)
            if (lastY != -1) {
                linePoints.add(Point(right.toDouble(), lastY.toDouble()))
            }

            if (linePoints.size > 4) {
                allLines.add(linePoints)
            }
        }

        val extractDuration = System.currentTimeMillis() - extractStart
        Log.v("ExtractControlPointsUtil", "  Point extraction: ${extractDuration}ms")

        val totalDuration = System.currentTimeMillis() - start
        Log.v("ExtractControlPointsUtil", "  Extract total: ${totalDuration}ms")

        labeled.release()
        stats.release()
        centroids.release()

        return allLines
    }


    private fun findSkeletonYFast(
        columnMap: HashMap<Int, MutableList<Int>>,
        x: Int,
        previousY: Int = -1
    ): Int {
        for (dx in -3..3) {
            val currX = x + dx
            val ysAtColumn = columnMap[currX] ?: continue

            if (ysAtColumn.isEmpty()) continue

            if (previousY == -1) return ysAtColumn[0]

            var bestY = ysAtColumn[0]
            var minDelta = kotlin.math.abs(bestY - previousY)

            for (y in ysAtColumn) {
                val delta = kotlin.math.abs(y - previousY)
                if (delta < minDelta) {
                    minDelta = delta
                    bestY = y
                }
            }
            return bestY
        }
        return -1
    }

    /**
     * Adaptive sampling for a discrete baseline represented by a columnMap.
     * Produces denser samples where curvature is high.
     */
    @Deprecated("Use extractControlPointsOptimized instead")
    private fun adaptiveSampleBaseline(
        sortedColumns: List<Int>,
        columnMap: Map<Int, List<Int>>,
        previousY: Int,
        minStep: Int = 3,
        maxStep: Int = 30,
        curvatureWindow: Int = 5
    ): List<Point> {

        fun yAt(x: Int): Double {
            val ys = columnMap[x]
            return if (ys == null || ys.isEmpty()) previousY.toDouble() else ys.average()
        }

        fun curvatureAt(x: Int): Double {
            val x0 = sortedColumns.first().coerceAtLeast(x - curvatureWindow)
            val x2 = sortedColumns.last().coerceAtMost(x + curvatureWindow)
            val y0 = yAt(x0)
            val y1 = yAt(x)
            val y2 = yAt(x2)
            return kotlin.math.abs(y2 - 2 * y1 + y0)
        }

        val result = mutableListOf<Point>()
        var x = sortedColumns.first()
        var lastY = yAt(x)
        result.add(Point(x.toDouble(), lastY))

        while (x < sortedColumns.last()) {
            val curv = curvatureAt(x)
            val step = (maxStep / (1.0 + curv * 10.0)).toInt().coerceIn(minStep, maxStep)
            x += step
            if (x > sortedColumns.last()) x = sortedColumns.last()

            val y = yAt(x)
            if (kotlin.math.abs(y - lastY) > 0.1) {
                result.add(Point(x.toDouble(), y))
                lastY = y
            }
        }

        return result
    }

}
