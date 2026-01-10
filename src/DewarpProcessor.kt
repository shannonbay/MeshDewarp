package com.github.shannonbay.rereader.util

import android.graphics.Bitmap
import androidx.core.graphics.createBitmap
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

class DewarpProcessor(
    private val yoloModel: YoloSegmentationModel
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
        // 1) Run YOLOv8 segmentation to get the 160x160 probability mask
        val probMask = yoloModel.predictMask(original)

        // 2) Convert Bitmap to OpenCV Mat
        val srcMat = Mat()
        Utils.bitmapToMat(original, srcMat)

        // 3) Create a binary mask Mat from the FloatArray (160x160)
        val maskMat = Mat(160, 160, CvType.CV_8UC1)
        val maskData = ByteArray(160 * 160)
        for (i in probMask.indices) {
            // Threshold: If probability > 0.5, set to 255 (white), else 0
            maskData[i] = if (probMask[i] > 0.5f) 255.toByte() else 0.toByte()
        }
        maskMat.put(0, 0, maskData)

        // 4) Find contours and draw the outline on the original image
        drawDocumentOutline(srcMat, maskMat)

        // 5) Use the mask to find the document corners and dewarp
        val dewarpedMat = applyDewarp(srcMat, probMask)

        // 6) Convert back to Bitmap
        val resultBitmap = createBitmap(dewarpedMat.cols(), dewarpedMat.rows())
        Utils.matToBitmap(dewarpedMat, resultBitmap)

        // Cleanup
        srcMat.release()
        maskMat.release()
        dewarpedMat.release()

        return resultBitmap
    }

    private fun drawDocumentOutline(srcMat: Mat, maskMat: Mat) {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()

        // Find contours on the 160x160 mask
        Imgproc.findContours(maskMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        // Find the largest contour (the document)
        val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) }

        largestContour?.let { contour ->
            // Scale contour points from 160x160 back to original image size
            val scaleX = srcMat.cols().toDouble() / 160.0
            val scaleY = srcMat.rows().toDouble() / 160.0

            val scaledPoints = contour.toArray().map { point ->
                Point(point.x * scaleX, point.y * scaleY)
            }

            val scaledContour = MatOfPoint().apply { fromArray(*scaledPoints.toTypedArray()) }

            // Draw a thick neon-green outline
            Imgproc.drawContours(srcMat, listOf(scaledContour), -1, Scalar(0.0, 255.0, 0.0, 255.0), 10)
        }
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