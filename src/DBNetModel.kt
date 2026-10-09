package com.github.shannonbay.rereader.util

import android.content.Context
import android.graphics.Bitmap
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.core.graphics.scale

class DBNetModel(context: Context) {

    private var interpreter: Interpreter
    private val inputSize = 960

    // Mean and Std for normalization (matches typical PP-OCR DBNet)
    private val mean = floatArrayOf(123.675f, 116.28f, 103.53f)
    private val std = floatArrayOf(58.395f, 57.12f, 57.375f)

    init {
        val options = Interpreter.Options().apply {
            // S21 handles GPU delegates very well for DBNet's Conv layers
            addDelegate(GpuDelegate())
            setNumThreads(4)
        }
        val modelBuffer = context.assets.open("ml/dbnet_det_fixed_float16.tflite").use {
            val bytes = it.readBytes()
            ByteBuffer.allocateDirect(bytes.size).apply {
                order(ByteOrder.nativeOrder())
                put(bytes)
            }
        }
        interpreter = Interpreter(modelBuffer, options)

        // Force the fixed shape for S21 NPU optimization
        interpreter.resizeInput(0, intArrayOf(1, inputSize, inputSize, 3))
        interpreter.allocateTensors()
    }

    fun getHeatmap(bitmap: Bitmap): Mat {
        val scaledBitmap = bitmap.scale(inputSize, inputSize)
        val inputBuffer = preprocess(scaledBitmap)

        // Output is 1x640x640x1
        val outputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 1 * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        // Convert the buffer to an OpenCV Mat for dewarping logic
        val heatmap = Mat(inputSize, inputSize, CvType.CV_32FC1)
        val floatArray = FloatArray(inputSize * inputSize)
        outputBuffer.asFloatBuffer().get(floatArray)
        heatmap.put(0, 0, floatArray)

        return heatmap
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            // Extract RGB and apply normalization
            buffer.putFloat(((pixel shr 16 and 0xFF) - mean[0]) / std[0])
            buffer.putFloat(((pixel shr 8 and 0xFF) - mean[1]) / std[1])
            buffer.putFloat(((pixel and 0xFF) - mean[2]) / std[2])
        }
        return buffer
    }

    fun close() {
        interpreter.close()
    }
}