package com.github.shannonbay.rereader.util

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

class YoloSegmentationModel(context: Context) {
    private val interpreter: Interpreter
    private val inputWidth = 640
    private val inputHeight = 640

    // Float16 models in TFLite still use 4-byte FloatArrays for I/O
    // Output 1: Detections [1, 37, 8400]
    private val output0 = Array(1) { Array(37) { FloatArray(8400) } }
    // Output 0: Prototypes [1, 160, 160, 32]
    private val output1 = Array(1) { Array(160) { Array(160) { FloatArray(32) } } }

    init {
        val options = Interpreter.Options().apply {
            try {
                // S21 GPU is much better for Float16 than NNAPI
                val gpuDelegate = org.tensorflow.lite.gpu.GpuDelegate()
                addDelegate(gpuDelegate)
            } catch (e: Exception) {
                android.util.Log.e("YOLO", "GPU Delegate failed, falling back to CPU", e)
                setNumThreads(4)
            }
        }

        val modelBuffer = context.assets.openFd("ml/doc_seg_yolov8n_float16.tflite").use {
            it.createInputStream().channel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY,
                it.startOffset,
                it.declaredLength
            )
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    fun predictMask(bitmap: Bitmap): FloatArray {
        val inputBuffer = preprocess(bitmap)

        // REVERSED MAPPING for the Float16 model:
        // Tensor 0: [1, 37, 8400]    -> output0
        // Tensor 1: [1, 160, 160, 32] -> output1
        val outputs = mutableMapOf<Int, Any>(
            0 to output0,
            1 to output1
        )

        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
        return postprocess()
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        // 4 bytes per Float * 3 channels * width * height
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputWidth * inputHeight * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(inputWidth * inputHeight)
        scaledBitmap.getPixels(intValues, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        byteBuffer.clear()
        for (pixelValue in intValues) {
            // Normalize RGB values to 0.0f - 1.0f
            byteBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f)
            byteBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)
            byteBuffer.putFloat((pixelValue and 0xFF) / 255.0f)
        }
        return byteBuffer
    }

    private fun postprocess(): FloatArray {
        var maxConf = -1f
        var bestIndex = -1

        // 1. Find the highest confidence detection in output0
        for (i in 0 until 8400) {
            val conf = output0[0][4][i]
            if (conf > maxConf) {
                maxConf = conf
                bestIndex = i
            }
        }

        android.util.Log.d("YOLO", "Float16 Best Confidence: $maxConf at index $bestIndex")

        // 2. Threshold check (0.4 - 0.5 is usually good for doc detection)
        if (bestIndex == -1 || maxConf < 0.45f) {
            return FloatArray(160 * 160) { 0f }
        }

        // 3. Extract 32 mask coefficients (indices 5 to 36)
        val coeffs = FloatArray(32)
        for (j in 0 until 32) {
            coeffs[j] = output0[0][j + 5][bestIndex]
        }

        // 4. Matrix multiplication: Coefficients x Prototype Masks
        val mask = FloatArray(160 * 160)
        for (y in 0 until 160) {
            for (x in 0 until 160) {
                var sum = 0f
                for (c in 0 until 32) {
                    sum += coeffs[c] * output1[0][y][x][c]
                }
                // Apply Sigmoid to get the 0.0-1.0 probability
                mask[y * 160 + x] = 1.0f / (1.0f + exp(-sum))
            }
        }
        return mask
    }
}