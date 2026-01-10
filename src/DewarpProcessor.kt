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

class DewarpProcessor(
    private val dbNetModel: DBNetModel
) {

    companion object {
        init {
            if (org.opencv.android.OpenCVLoader.initDebug()) {
                android.util.Log.i("OpenCV", "OpenCV loaded successfully")
            } else {
                android.util.Log.e("OpenCV", "OpenCV initialization failed!")
            }
        }
    }

    /**
     * Processes the input bitmap to flatten the page using YOLO segmentation.
     */
    fun process(original: Bitmap): Bitmap {
        val probMask = dbNetModel.getHeatmap(original)

        // 1. Threshold to Binary (0 or 255)
        val binary = Mat()
        // DBNet usually needs a threshold around 0.3 for clean lines
        Imgproc.threshold(probMask, binary, 0.3, 255.0, Imgproc.THRESH_BINARY)
        binary.convertTo(binary, CvType.CV_8UC1)

        // 2. Skeletonize (Thinning)
        val skeleton = skeletonize(binary)

        val linePoints = extractSplineControlPoints(skeleton)
        val splicedLinePoints = spliceLineFragments(linePoints)

        val splicedLines = drawSplicedLines(skeleton, splicedLinePoints)

        // 3. Optional: Extract Points for Splines
        // val splinePoints = extractSplinePoints(skeleton)

        val resultBitmap = createBitmap(splicedLines.cols(), splicedLines.rows())
        org.opencv.android.Utils.matToBitmap(splicedLines, resultBitmap)

        // Cleanup
        listOf(splicedLines, probMask, binary, skeleton).forEach { it.release() }
        return resultBitmap
    }

    /**
     * Overlays spliced lines onto a gray Mat for visual inspection.
     * @param grayBase The 960x960 CV_8UC1 Mat (thresholded or skeletonized)
     * @param splicedLines The list of point-lists from our splicing logic
     * @return A CV_8UC4 (RGBA) Mat with color-coded lines
     */
    fun drawSplicedLines(grayBase: Mat, splicedLines: List<List<Point>>): Mat {
        val debugMat = Mat()

        // 1. Convert to RGBA so we can draw in color
        Imgproc.cvtColor(grayBase, debugMat, Imgproc.COLOR_GRAY2RGBA)

        // 2. Slightly dim the background to make lines stand out
        // (Alpha 0.4 reduces contrast, Beta 50 adds a gray floor)
        debugMat.convertTo(debugMat, -1, 0.4, 50.0)

        val colors = listOf(
            Scalar(255.0, 50.0, 50.0, 255.0),   // Bright Red
            Scalar(50.0, 255.0, 50.0, 255.0),   // Bright Green
            Scalar(80.0, 150.0, 255.0, 255.0),  // Sky Blue
            Scalar(255.0, 255.0, 50.0, 255.0),  // Yellow
            Scalar(255.0, 50.0, 255.0, 255.0)   // Magenta
        )

        // 3. Draw the paths
        splicedLines.forEachIndexed { index, points ->
            val color = colors[index % colors.size]

            for (i in 0 until points.size - 1) {
                // Draw the connection
                Imgproc.line(debugMat, points[i], points[i + 1], color, 2, Imgproc.LINE_AA)

                // Draw the control point "anchor"
                Imgproc.circle(debugMat, points[i], 3, color, -1)
            }

            // Mark the final point of the line
            if (points.isNotEmpty()) {
                Imgproc.circle(debugMat, points.last(), 3, color, -1)
            }
        }

        return debugMat
    }

    /**
     * Zhang-Suen Thinning Algorithm
     * Reduces binary blobs to 1-pixel wide centerlines.
     */
    private fun skeletonize(input: Mat): Mat {
        val res = input.clone()

        // Iterative erosion/opening until the image stops changing
        // This is a fast approximation of the medial axis
        val element = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, Size(3.0, 3.0))
        val skel = Mat.zeros(res.size(), res.type())
        val temp = Mat()
        val eroded = Mat()

        var done = false
        while (!done) {
            Imgproc.erode(res, eroded, element)
            Imgproc.dilate(eroded, temp, element)
            Core.subtract(res, temp, temp)
            Core.bitwise_or(skel, temp, skel)
            eroded.copyTo(res)

            if (Core.countNonZero(res) == 0) done = true
        }
        return skel
    }

    /**
     * Extracts ordered control points from the skeletonized Mat.
     * Returns a List of Lines, where each Line is a List of Points.
     */
    fun extractSplineControlPoints(skeleton: Mat): List<List<Point>> {
        val labeled = Mat()
        val stats = Mat()
        val centroids = Mat()

        // 1. Group skeleton pixels into individual text lines
        val numLabels = Imgproc.connectedComponentsWithStats(skeleton, labeled, stats, centroids)
        val allLines = mutableListOf<List<Point>>()

        // Start from 1 (0 is background)
        for (i in 1 until numLabels) {
            val linePoints = mutableListOf<Point>()
            val left = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val width = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val right = left + width - 1 // The absolute last pixel

            // 1. ALWAYS sample the very first pixel on the left
            val firstY = findSkeletonY(skeleton, labeled, i, left)
            if (firstY != -1) linePoints.add(Point(left.toDouble(), firstY.toDouble()))

            // 2. Sample the middle at regular intervals
            val step = 20 // Reduced step for higher density
            for (x in (left + step) until (right - step) step step) {
                val y = findSkeletonY(skeleton, labeled, i, x)
                if (y != -1) linePoints.add(Point(x.toDouble(), y.toDouble()))
            }

            // 3. ALWAYS sample the very last pixel on the right
            val lastY = findSkeletonY(skeleton, labeled, i, right)
            if (lastY != -1) linePoints.add(Point(right.toDouble(), lastY.toDouble()))

            if (linePoints.size > 4) { // Need at least 4 points for a cubic spline
                allLines.add(linePoints)
            }
        }
        return allLines
    }

    private fun findSkeletonY(
        skeleton: Mat,
        labels: Mat,
        targetLabel: Int,
        x: Int,
        previousY: Int = -1
    ): Int {
        val rows = skeleton.rows()
        val cols = skeleton.cols()

        var bestY = -1
        var minDeltaY = Int.MAX_VALUE

        // Search a 7-pixel horizontal neighborhood (x-3 to x+3)
        for (dx in -3..3) {
            val currX = x + dx
            if (currX < 0 || currX >= cols) continue

            for (y in 0 until rows) {
                if (labels.get(y, currX)[0].toInt() == targetLabel) {
                    // If this is our first point in the line, any Y will do
                    if (previousY == -1) return y

                    // Otherwise, pick the Y closest to our last sampled point
                    // to prevent "jumping" to noise or loops
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

    data class SpliceCandidate(
        val leftIdx: Int,
        val rightIdx: Int,
        val cost: Double
    )

    fun spliceLineFragments(
        fragments: List<List<Point>>,
        xGapThreshold: Double = 180.0
    ): List<List<Point>> {
        val TAG = "SpliceContest"
        val lines = fragments.toMutableList()
        android.util.Log.d(TAG, "Starting splice with ${lines.size} fragments")

        var totalJoins = 0

        while (true) {
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

                        if (cost < 80.0) { // Slightly relaxed gate for testing
                            candidates.add(SpliceCandidate(i, j, cost))
                        } else {
                            // Log why we didn't even consider it
                            // android.util.Log.v(TAG, "Rejected candidate i:$i->j:$j - Cost too high: $cost")
                        }
                    }
                }
            }

            if (candidates.isEmpty()) {
                android.util.Log.d(TAG, "No more candidates found. Splice complete.")
                break
            }

            candidates.sortBy { it.cost }

            var joinedInThisIteration = false
            for (best in candidates) {
                val left = lines[best.leftIdx]
                val right = lines[best.rightIdx]

                // HARD GUARD: Does this bridge physically cross ANY other line?
                if (doesBridgeCrossExistingLines(left, right, lines)) {
                    android.util.Log.w("SpliceContest", "REJECTED: Bridge crosses existing text line.")
                    continue
                }

                // SOFT GUARD: Is the corridor blocked? (Your previous segment-distance logic)
                if (isLineIntervening(left, right, lines)) {
                    continue
                }

                // If it passes both, it's a safe join
                val merged = left + right
                val toRemove = listOf(left, right)
                lines.removeAll { it in toRemove }
                lines.add(merged)

                totalJoins++
                android.util.Log.i(TAG, "JOIN SUCCESS: Cost ${"%.2f".format(best.cost)} | Gap: ${"%.1f".format(right.first().x - left.last().x)}px")
                joinedInThisIteration = true
                break // Break to re-index and re-calculate candidates
            }

            if (!joinedInThisIteration) {
                android.util.Log.d(TAG, "Candidates existed but all were blocked by interventions.")
                break
            }
        }

        android.util.Log.d(TAG, "Final result: ${lines.size} lines (Total joins: $totalJoins)")
        return lines
    }

    private fun calculateSpliceCost(left: List<Point>, right: List<Point>, gap: Double): Double {
        val yDist = Math.abs(left.last().y - right.first().y)

        // TINY GAP CASE: If they are basically touching horizontally
        if (gap < 5.0) {
            // Just return the vertical distance. If it's less than 10px, it's a perfect match.
            return yDist
        }

        val exitSlope = calculateExitSlope(left)
        val entrySlope = calculateEntrySlope(right)
        val bridgeSlope = (right.first().y - left.last().y) / gap

        // Standard projection error
        val projY = left.last().y + (exitSlope * gap)
        val verticalDist = Math.abs(projY - right.first().y)

        // Angular error
        val slopeMatch = Math.abs(exitSlope - entrySlope)
        val bridgeMatch = Math.abs(exitSlope - bridgeSlope)

        // Combine. We use a lower multiplier for bridgeMatch here
        // because steep curves in the gutter are expected.
        return verticalDist + (slopeMatch * 40.0) + (bridgeMatch * 40.0)
    }

    // Helper to ensure we aren't jumping over another line segment
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

        // Head-to-tail fragments don't need intervention checks
        if (gapWidth < 8.0) return false

        // Shrink the horizontal check zone to ignore the immediate tips
        val checkAreaStart = xStart + (gapWidth * 0.1)
        val checkAreaEnd = xEnd - (gapWidth * 0.1)

        val vTolerance = 7.0 // Allowed distance from the diagonal bridge

        for (line in allLines) {
            if (line === left || line === right) continue

            for (p in line) {
                // Only check points within the horizontal span of the gap
                if (p.x > checkAreaStart && p.x < checkAreaEnd) {
                    // Calculate distance from point P to the segment (p1, p2)
                    val dist = distanceToSegment(p, p1, p2)
                    if (dist < vTolerance) {
                        return true // Something is physically blocking the diagonal path
                    }
                }
            }
        }
        return false
    }

    /**
     * Calculates the perpendicular distance from point P to line segment AB
     */
    private fun distanceToSegment(p: Point, a: Point, b: Point): Double {
        val l2 = Math.pow(b.x - a.x, 2.0) + Math.pow(b.y - a.y, 2.0)
        if (l2 == 0.0) return Math.sqrt(Math.pow(p.x - a.x, 2.0) + Math.pow(p.y - a.y, 2.0))

        // Projection of P onto the line segment
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
            // Skip the two fragments we are currently joining
            if (line === left || line === right) continue

            // Check every segment of every other line
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

    /**
     * Standard CCW (Counter-Clockwise) segment intersection test
     */
    private fun doSegmentsIntersect(p1: Point, p2: Point, p3: Point, p4: Point): Boolean {
        fun ccw(a: Point, b: Point, c: Point): Boolean {
            return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)
        }
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) && ccw(p1, p2, p3) != ccw(p1, p2, p4)
    }

    private fun calculateExitSlope(points: List<Point>): Double {
        if (points.size < 5) return 0.0
        // Use the last 5 points to get a stable local slope
        val sample = points.takeLast(5)
        val dx = sample.last().x - sample.first().x
        val dy = sample.last().y - sample.first().y
        return if (dx != 0.0) dy / dx else 0.0
    }

    /**
     * Calculates the trajectory of the start of a line fragment.
     * Uses the first 5 points to determine the "entry angle".
     */
    private fun calculateEntrySlope(points: List<Point>): Double {
        if (points.size < 5) return 0.0

        // Use the first 5 points to get a stable local slope
        val sample = points.take(5)
        val dx = sample.last().x - sample.first().x
        val dy = sample.last().y - sample.first().y

        return if (dx != 0.0) dy / dx else 0.0
    }

    private fun applyDewarp(src: Mat, mask: FloatArray): Mat {
        // Placeholder for the OpenCV logic:
        // 1. Threshold the 160x160 mask to binary
        // 2. Find external contours
        // 3. Find 4 corners (approxPolyDP)
        // 4. Imgproc.getPerspectiveTransform & Imgproc.warpPerspective
        return src // Currently returns original; logic to be implemented
    }
}