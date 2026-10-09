package com.github.shannonbay.rereader.util

import android.util.Log
import org.opencv.core.Point

/**
 * Utility class for splicing line fragments that belong to the same text line.
 * Uses trajectory matching, projection error, and intervention checking
 * to determine which fragments should be joined.
 */
object SplineSplicer {

    private const val TAG = "SplineSplicer"

    data class SpliceCandidate(
        val leftIdx: Int,
        val rightIdx: Int,
        val cost: Double
    )

    /**
     * Splices line fragments that belong to the same text line.
     * Uses trajectory matching, projection error, and intervention checking
     * to determine which fragments should be joined.
     */
    fun spliceLineFragments(
        fragments: List<List<Point>>,
        xGapThreshold: Double = 180.0
    ): List<List<Point>> {
        val lines = fragments.toMutableList()
        Log.d(TAG, "Starting splice with ${lines.size} fragments")

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

                        if (cost < 80.0) {
                            candidates.add(SpliceCandidate(i, j, cost))
                        }
                    }
                }
            }

            if (candidates.isEmpty()) {
                Log.d(TAG, "No more splice candidates found. Splice complete.")
                break
            }

            candidates.sortBy { it.cost }

            var joinedInThisIteration = false
            for (best in candidates) {
                val left = lines[best.leftIdx]
                val right = lines[best.rightIdx]

                // HARD GUARD: Does this bridge physically cross ANY other line?
                if (doesBridgeCrossExistingLines(left, right, lines)) {
                    Log.w(TAG, "REJECTED: Bridge crosses existing text line.")
                    continue
                }

                // SOFT GUARD: Is the corridor blocked?
                if (isLineIntervening(left, right, lines)) {
                    continue
                }

                // If it passes both, it's a safe join
                val merged = left + right
                val toRemove = listOf(left, right)
                lines.removeAll { it in toRemove }
                lines.add(merged)

                totalJoins++
                Log.i(TAG, "JOIN SUCCESS: Cost ${"%.2f".format(best.cost)} | Gap: ${"%.1f".format(right.first().x - left.last().x)}px")
                joinedInThisIteration = true
                break // Break to re-index and re-calculate candidates
            }

            if (!joinedInThisIteration) {
                Log.d(TAG, "Candidates existed but all were blocked by interventions.")
                break
            }
        }

        Log.d(TAG, "Final result: ${lines.size} lines (Total joins: $totalJoins)")
        return lines
    }

    private fun calculateSpliceCost(left: List<Point>, right: List<Point>, gap: Double): Double {
        val yDist = Math.abs(left.last().y - right.first().y)

        // TINY GAP CASE: If they are basically touching horizontally
        if (gap < 5.0) {
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

        // Head-to-tail fragments don't need intervention checks
        if (gapWidth < 8.0) return false

        // Shrink the horizontal check zone to ignore the immediate tips
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
}