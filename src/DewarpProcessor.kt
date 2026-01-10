package com.github.shannonbay.rereader.util

import android.graphics.Bitmap
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import androidx.core.graphics.createBitmap
import org.opencv.core.Core
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

        // 3. Optional: Extract Points for Splines
        // val splinePoints = extractSplinePoints(skeleton)

        // Visualization: Show the skeleton in green over a gray version of the page
        val resultMat = Mat()
        Imgproc.cvtColor(skeleton, resultMat, Imgproc.COLOR_GRAY2RGBA)
        // resultMat.setTo(Scalar(0.0, 255.0, 0.0), skeleton) // Debug: Show skeleton in green

        val resultBitmap = createBitmap(resultMat.cols(), resultMat.rows())
        org.opencv.android.Utils.matToBitmap(resultMat, resultBitmap)

        // Cleanup
        listOf(probMask, binary, skeleton, resultMat).forEach { it.release() }
        return resultBitmap
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

    private fun applyDewarp(src: Mat, mask: FloatArray): Mat {
        // Placeholder for the OpenCV logic:
        // 1. Threshold the 160x160 mask to binary
        // 2. Find external contours
        // 3. Find 4 corners (approxPolyDP)
        // 4. Imgproc.getPerspectiveTransform & Imgproc.warpPerspective
        return src // Currently returns original; logic to be implemented
    }
}