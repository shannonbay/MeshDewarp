package com.github.shannonbay.rereader.util

import android.graphics.Bitmap
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import androidx.core.graphics.createBitmap

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
        // 1) Run DBNet to get the 960x960 probability map (CV_32FC1)
        val probMask = dbNetModel.getHeatmap(original)

        // 2) Convert from Float (0.0-1.0) to UByte (0-255)
        val grayMat = Mat()
        probMask.convertTo(grayMat, CvType.CV_8UC1, 255.0)

        // 3) OPTIONAL: Apply a color map so you can actually see the "S-curves"
        val colorMat = Mat()
        Imgproc.applyColorMap(grayMat, colorMat, Imgproc.COLORMAP_JET)

        // 4) Convert back to Bitmap for display in rereader
        // Ensure resultBitmap is initialized to the correct size (640x640)
        val resultBitmap = createBitmap(colorMat.cols(), colorMat.rows())
        org.opencv.android.Utils.matToBitmap(colorMat, resultBitmap)

        // Cleanup Mats to prevent memory leaks in the core loop
        probMask.release()
        grayMat.release()
        colorMat.release()

        return resultBitmap
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