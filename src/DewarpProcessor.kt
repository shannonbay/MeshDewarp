package com.github.shannonbay.rereader.util

import android.graphics.Bitmap
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import androidx.core.graphics.createBitmap
import org.opencv.core.Core
import org.opencv.core.Point
import org.opencv.core.Scalar
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
     * Processes the input bitmap to flatten the page using YOLO segmentation.
     */
    fun process(original: Bitmap): Bitmap {
        val totalStart = System.currentTimeMillis()
        Log.d(TAG, "========== Starting Dewarp Pipeline ==========")
        Log.d(TAG, "Input size: ${original.width}x${original.height}")

        // Stage 1: DBNet Heatmap Generation
        var stageStart = System.currentTimeMillis()
        val probMask = dbNetModel.getHeatmap(original)
        var stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[1/8] DBNet Heatmap: ${stageDuration}ms")

        // Stage 2: Thresholding to Binary
        stageStart = System.currentTimeMillis()
        val binary = Mat()
        Imgproc.threshold(probMask, binary, 0.3, 255.0, Imgproc.THRESH_BINARY)
        binary.convertTo(binary, CvType.CV_8UC1)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[2/8] Thresholding: ${stageDuration}ms")

        // Stage 3: Skeletonization
        stageStart = System.currentTimeMillis()
        val skeleton = skeletonize(binary)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[3/8] Skeletonization: ${stageDuration}ms")

        // Stage 4: Bridge Gaps - OPTIMIZED
        stageStart = System.currentTimeMillis()
        bridgeSkeletonGapsOptimized(skeleton)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[4/8] Bridge Gaps (Optimized): ${stageDuration}ms")

        // Stage 5: Morphological Close
        stageStart = System.currentTimeMillis()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 1.0))
        Imgproc.morphologyEx(skeleton, skeleton, Imgproc.MORPH_CLOSE, kernel)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[5/8] Morphological Close: ${stageDuration}ms")

        // Stage 6: Extract Control Points - OPTIMIZED
        stageStart = System.currentTimeMillis()
        val linePoints = extractSplineControlPointsOptimized(skeleton)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[6/8] Extract Control Points (Optimized): ${stageDuration}ms (${linePoints.size} fragments)")

        // Stage 7: Splice Fragments
        stageStart = System.currentTimeMillis()
        val splicedLinePoints = spliceLineFragments(linePoints)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[7/8] Splice Fragments: ${stageDuration}ms (${linePoints.size} -> ${splicedLinePoints.size} lines)")

        // Stage 8: Draw Result
        stageStart = System.currentTimeMillis()
        val splicedLines = drawSplicedLines(skeleton, splicedLinePoints)
        val resultBitmap = createBitmap(splicedLines.cols(), splicedLines.rows())
        org.opencv.android.Utils.matToBitmap(splicedLines, resultBitmap)
        stageDuration = System.currentTimeMillis() - stageStart
        Log.d(TAG, "[8/8] Draw & Convert: ${stageDuration}ms")

        // Cleanup
        listOf(splicedLines, probMask, binary, skeleton).forEach { it.release() }

        val totalDuration = System.currentTimeMillis() - totalStart
        Log.d(TAG, "========== Pipeline Complete: ${totalDuration}ms ==========")

        return resultBitmap
    }

    /**
     * OPTIMIZED: Uses Connected Components to find terminals + downsampling
     */
    fun bridgeSkeletonGapsOptimized(skeleton: Mat, maxGap: Double = 5.0) {
        val start = System.currentTimeMillis()

        // Downsample for faster terminal detection (4x smaller = 16x fewer pixels)
        val downscaleFactor = 4.0
        val small = Mat()
        Imgproc.resize(skeleton, small, Size(), 1.0/downscaleFactor, 1.0/downscaleFactor, Imgproc.INTER_NEAREST)

        // Run Connected Components on downsampled image
        val labeled = Mat()
        val stats = Mat()
        val centroids = Mat()
        val findStart = System.currentTimeMillis()
        val numLabels = Imgproc.connectedComponentsWithStats(small, labeled, stats, centroids)

        // Convert to array for fast access
        val rows = labeled.rows()
        val cols = labeled.cols()
        val labelData = IntArray(rows * cols)
        labeled.get(0, 0, labelData)

        // Spatial hash for terminals
        val cellSize = ((maxGap / downscaleFactor) * 2).toInt().coerceAtLeast(2)
        val gridCols = (cols + cellSize - 1) / cellSize
        val spatialGrid = HashMap<Int, MutableList<Point>>()

        fun getGridKey(x: Int, y: Int): Int {
            val gx = x / cellSize
            val gy = y / cellSize
            return gy * gridCols + gx
        }

        // Find terminal points using CC-based approach
        var terminalCount = 0
        for (label in 1 until numLabels) {
            val componentPixels = mutableListOf<Point>()

            // Collect all pixels in this component
            for (y in 0 until rows) {
                for (x in 0 until cols) {
                    if (labelData[y * cols + x] == label) {
                        componentPixels.add(Point(x.toDouble(), y.toDouble()))
                    }
                }
            }

            // For each pixel, check if it's a terminal (1 neighbor in same component)
            for (p in componentPixels) {
                val x = p.x.toInt()
                val y = p.y.toInt()

                var neighborCount = 0
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        val nx = x + dx
                        val ny = y + dy
                        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                            if (labelData[ny * cols + nx] == label) {
                                neighborCount++
                            }
                        }
                    }
                }

                if (neighborCount == 1) {
                    // Scale back to full resolution
                    val fullResPoint = Point(p.x * downscaleFactor, p.y * downscaleFactor)
                    val key = getGridKey(x, y)
                    spatialGrid.getOrPut(key) { mutableListOf() }.add(fullResPoint)
                    terminalCount++
                }
            }
        }

        val findDuration = System.currentTimeMillis() - findStart
        Log.v(TAG, "  Found $terminalCount terminal points (${findDuration}ms, downsampled ${downscaleFactor}x)")

        // Connect near-neighbors using spatial hash
        val connectStart = System.currentTimeMillis()
        var bridgeCount = 0

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

                            val dist = Math.sqrt(
                                Math.pow(p1.x - p2.x, 2.0) + Math.pow(p1.y - p2.y, 2.0)
                            )

                            if (dist > 0 && dist <= maxGap) {
                                Imgproc.line(skeleton, p1, p2, Scalar(255.0), 1)
                                bridgeCount++
                            }
                        }
                    }
                }
            }
        }

        val connectDuration = System.currentTimeMillis() - connectStart
        Log.v(TAG, "  Created $bridgeCount bridges (${connectDuration}ms)")

        // Cleanup
        small.release()
        labeled.release()
        stats.release()
        centroids.release()

        val totalDuration = System.currentTimeMillis() - start
        Log.v(TAG, "  Bridge total: ${totalDuration}ms")
    }

    /**
     * OPTIMIZED: Direct pixel access instead of repeated Mat.get() calls
     */
    fun extractSplineControlPointsOptimized(skeleton: Mat): List<List<Point>> {
        val start = System.currentTimeMillis()
        val labeled = Mat()
        val stats = Mat()
        val centroids = Mat()

        // Connected components analysis
        val ccStart = System.currentTimeMillis()
        val numLabels = Imgproc.connectedComponentsWithStats(skeleton, labeled, stats, centroids)
        val ccDuration = System.currentTimeMillis() - ccStart
        Log.v(TAG, "  Connected Components: $numLabels labels (${ccDuration}ms)")

        val allLines = mutableListOf<List<Point>>()
        val extractStart = System.currentTimeMillis()

        // Convert labeled Mat to array for faster access
        val rows = labeled.rows()
        val cols = labeled.cols()
        val labelData = IntArray(rows * cols)
        labeled.get(0, 0, labelData)

        for (i in 1 until numLabels) {
            val linePoints = mutableListOf<Point>()
            val left = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val width = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val right = left + width - 1

            // Sample at regular intervals with optimized search
            val step = 20
            var previousY = -1

            // First point
            val firstY = findSkeletonYOptimized(labelData, cols, rows, i, left, previousY)
            if (firstY != -1) {
                linePoints.add(Point(left.toDouble(), firstY.toDouble()))
                previousY = firstY
            }

            // Middle points
            for (x in (left + step) until (right - step) step step) {
                val y = findSkeletonYOptimized(labelData, cols, rows, i, x, previousY)
                if (y != -1) {
                    linePoints.add(Point(x.toDouble(), y.toDouble()))
                    previousY = y
                }
            }

            // Last point
            val lastY = findSkeletonYOptimized(labelData, cols, rows, i, right, previousY)
            if (lastY != -1) {
                linePoints.add(Point(right.toDouble(), lastY.toDouble()))
            }

            if (linePoints.size > 4) {
                allLines.add(linePoints)
            }
        }

        val extractDuration = System.currentTimeMillis() - extractStart
        Log.v(TAG, "  Point extraction: ${extractDuration}ms")

        val totalDuration = System.currentTimeMillis() - start
        Log.v(TAG, "  Extract total: ${totalDuration}ms")

        return allLines
    }

    /**
     * OPTIMIZED: Uses direct array access instead of Mat.get()
     */
    private fun findSkeletonYOptimized(
        labelData: IntArray,
        cols: Int,
        rows: Int,
        targetLabel: Int,
        x: Int,
        previousY: Int = -1
    ): Int {
        var bestY = -1
        var minDeltaY = Int.MAX_VALUE

        // Search a 7-pixel horizontal neighborhood
        for (dx in -3..3) {
            val currX = x + dx
            if (currX < 0 || currX >= cols) continue

            for (y in 0 until rows) {
                val index = y * cols + currX
                if (labelData[index] == targetLabel) {
                    if (previousY == -1) return y

                    val deltaY = Math.abs(y - previousY)
                    if (deltaY < minDeltaY) {
                        minDeltaY = deltaY
                        bestY = y
                    }
                }
            }
        }
        return bestY
    }

    /**
     * Overlays spliced lines onto a gray Mat for visual inspection.
     */
    fun drawSplicedLines(grayBase: Mat, splicedLines: List<List<Point>>): Mat {
        val debugMat = Mat()

        Imgproc.cvtColor(grayBase, debugMat, Imgproc.COLOR_GRAY2RGBA)
        debugMat.convertTo(debugMat, -1, 0.4, 50.0)

        val colors = listOf(
            Scalar(255.0, 50.0, 50.0, 255.0),
            Scalar(50.0, 255.0, 50.0, 255.0),
            Scalar(80.0, 150.0, 255.0, 255.0),
            Scalar(255.0, 255.0, 50.0, 255.0),
            Scalar(255.0, 50.0, 255.0, 255.0)
        )

        splicedLines.forEachIndexed { index, points ->
            val color = colors[index % colors.size]

            for (i in 0 until points.size - 1) {
                Imgproc.line(debugMat, points[i], points[i + 1], color, 2, Imgproc.LINE_AA)
                Imgproc.circle(debugMat, points[i], 3, color, -1)
            }

            if (points.isNotEmpty()) {
                Imgproc.circle(debugMat, points.last(), 3, color, -1)
            }
        }

        return debugMat
    }

    /**
     * Zhang-Suen Thinning Algorithm
     */
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
        Log.v(TAG, "  Skeletonize iterations: $iterations (${duration}ms)")

        return skel
    }

    private fun countNeighbors(mat: Mat, x: Int, y: Int): Int {
        var count = 0
        for (dy in -1..1) {
            for (dx in -1..1) {
                if (dx == 0 && dy == 0) continue
                if (mat.get(y + dy, x + dx)[0] > 0) count++
            }
        }
        return count
    }

    data class SpliceCandidate(
        val leftIdx: Int,
        val rightIdx: Int,
        val cost: Double
    )

    fun spliceLineFragments(
        fragments: List<List<Point>>,
        xGapThreshold: Double = 180.0
    ): List<List<Point>> {
        val spliceTag = "SpliceContest"
        val start = System.currentTimeMillis()
        val lines = fragments.toMutableList()
        Log.d(spliceTag, "Starting splice with ${lines.size} fragments")

        var totalJoins = 0
        var iteration = 0
        var candidateSearchTime = 0L
        var intersectionTestTime = 0L

        while (true) {
            iteration++
            val iterStart = System.currentTimeMillis()

            // Find candidates
            val candStart = System.currentTimeMillis()
            val candidates = mutableListOf<SpliceCandidate>()

            for (i in 0 until lines.size) {
                for (j in 0 until lines.size) {
                    if (i == j) continue
                    val lineA = lines[i]
                    val lineB = lines[j]

                    if (lineA.last().x >= lineB.first().x) continue

                    val gapX = lineB.first().x - lineA.last().x
                    if (gapX > 0 && gapX < xGapThreshold) {
                        val cost = calculateSpliceCost(lineA, lineB, gapX)

                        if (cost < 80.0) {
                            candidates.add(SpliceCandidate(i, j, cost))
                        }
                    }
                }
            }
            val candDuration = System.currentTimeMillis() - candStart
            candidateSearchTime += candDuration

            if (candidates.isEmpty()) {
                Log.d(spliceTag, "No more candidates found. Splice complete.")
                break
            }

            candidates.sortBy { it.cost }

            var joinedInThisIteration = false
            val testStart = System.currentTimeMillis()

            for (best in candidates) {
                val left = lines[best.leftIdx]
                val right = lines[best.rightIdx]

                if (doesBridgeCrossExistingLines(left, right, lines)) {
                    Log.w(spliceTag, "REJECTED: Bridge crosses existing text line.")
                    continue
                }

                if (isLineIntervening(left, right, lines)) {
                    continue
                }

                val merged = left + right
                val toRemove = listOf(left, right)
                lines.removeAll { it in toRemove }
                lines.add(merged)

                totalJoins++
                Log.i(spliceTag, "JOIN SUCCESS [Iter $iteration]: Cost ${"%.2f".format(best.cost)} | Gap: ${"%.1f".format(right.first().x - left.last().x)}px")
                joinedInThisIteration = true
                break
            }

            val testDuration = System.currentTimeMillis() - testStart
            intersectionTestTime += testDuration

            val iterDuration = System.currentTimeMillis() - iterStart
            Log.v(spliceTag, "  Iteration $iteration: ${iterDuration}ms (${candidates.size} candidates)")

            if (!joinedInThisIteration) {
                Log.d(spliceTag, "Candidates existed but all were blocked by interventions.")
                break
            }
        }

        val totalDuration = System.currentTimeMillis() - start
        Log.d(spliceTag, "Final result: ${lines.size} lines (Total joins: $totalJoins)")
        Log.d(spliceTag, "Splice timing breakdown:")
        Log.d(spliceTag, "  Candidate search: ${candidateSearchTime}ms")
        Log.d(spliceTag, "  Intersection tests: ${intersectionTestTime}ms")
        Log.d(spliceTag, "  Total splice time: ${totalDuration}ms")
        Log.d(spliceTag, "  Iterations: $iteration")

        return lines
    }

    private fun calculateSpliceCost(left: List<Point>, right: List<Point>, gap: Double): Double {
        val yDist = Math.abs(left.last().y - right.first().y)

        if (gap < 5.0) {
            return yDist
        }

        val exitSlope = calculateExitSlope(left)
        val entrySlope = calculateEntrySlope(right)
        val bridgeSlope = (right.first().y - left.last().y) / gap

        val projY = left.last().y + (exitSlope * gap)
        val verticalDist = Math.abs(projY - right.first().y)

        val slopeMatch = Math.abs(exitSlope - entrySlope)
        val bridgeMatch = Math.abs(exitSlope - bridgeSlope)

        return verticalDist + (slopeMatch * 40.0) + (bridgeMatch * 40.0)
    }

    private fun isLineIntervening(
        left: List<Point>,
        right: List<Point>,
        allLines: List<List<Point>>
    ): Boolean {
        val p1 = left.last()
        val p2 = right.first()

        val xStart = p1.x
        val xEnd = p2.x
        val gapWidth = xEnd - xStart

        if (gapWidth < 8.0) return false

        val checkAreaStart = xStart + (gapWidth * 0.1)
        val checkAreaEnd = xEnd - (gapWidth * 0.1)

        val vTolerance = 7.0

        for (line in allLines) {
            if (line === left || line === right) continue

            for (p in line) {
                if (p.x > checkAreaStart && p.x < checkAreaEnd) {
                    val dist = distanceToSegment(p, p1, p2)
                    if (dist < vTolerance) {
                        return true
                    }
                }
            }
        }
        return false
    }

    private fun distanceToSegment(p: Point, a: Point, b: Point): Double {
        val l2 = Math.pow(b.x - a.x, 2.0) + Math.pow(b.y - a.y, 2.0)
        if (l2 == 0.0) return Math.sqrt(Math.pow(p.x - a.x, 2.0) + Math.pow(p.y - a.y, 2.0))

        var t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2
        t = Math.max(0.0, Math.min(1.0, t))

        val projX = a.x + t * (b.x - a.x)
        val projY = a.y + t * (b.y - a.y)

        return Math.sqrt(Math.pow(p.x - projX, 2.0) + Math.pow(p.y - projY, 2.0))
    }

    private fun doesBridgeCrossExistingLines(
        left: List<Point>,
        right: List<Point>,
        allLines: List<List<Point>>
    ): Boolean {
        val bridgeA = left.last()
        val bridgeB = right.first()

        for (line in allLines) {
            if (line === left || line === right) continue

            for (i in 0 until line.size - 1) {
                val wallA = line[i]
                val wallB = line[i + 1]

                if (doSegmentsIntersect(bridgeA, bridgeB, wallA, wallB)) {
                    return true
                }
            }
        }
        return false
    }

    private fun doSegmentsIntersect(p1: Point, p2: Point, p3: Point, p4: Point): Boolean {
        fun ccw(a: Point, b: Point, c: Point): Boolean {
            return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)
        }
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) && ccw(p1, p2, p3) != ccw(p1, p2, p4)
    }

    private fun calculateExitSlope(points: List<Point>): Double {
        if (points.size < 5) return 0.0
        val sample = points.takeLast(5)
        val dx = sample.last().x - sample.first().x
        val dy = sample.last().y - sample.first().y
        return if (dx != 0.0) dy / dx else 0.0
    }

    private fun calculateEntrySlope(points: List<Point>): Double {
        if (points.size < 5) return 0.0

        val sample = points.take(5)
        val dx = sample.last().x - sample.first().x
        val dy = sample.last().y - sample.first().y

        return if (dx != 0.0) dy / dx else 0.0
    }

    private fun applyDewarp(src: Mat, mask: FloatArray): Mat {
        return src
    }
}