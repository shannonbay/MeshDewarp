package com.github.shannonbay.rereader.util

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.opencv.android.Utils
import org.opencv.core.Mat
import androidx.core.graphics.createBitmap

class ImageUtils {

    companion object {
        private const val TAG = "ImageUtils"

        /**
         * Saves a bitmap to the device gallery for inspection.
         * Pass the lifecycleScope from the Fragment to handle the Toast on the Main thread.
         */
        fun saveBitmapToGallery(
            context: Context,
            scope: CoroutineScope,
            bitmap: Bitmap,
            filename: String
        ) {
            try {
                val contentValues = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/ReReader")
                }

                val resolver = context.contentResolver
                val imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

                imageUri?.let { uri ->
                    resolver.openOutputStream(uri)?.use { outputStream ->
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, outputStream)
                        Log.d(TAG, "Saved image to gallery: $uri")

                        // Notify user on main thread using the passed scope
                        scope.launch(Dispatchers.Main) {
                            Toast.makeText(
                                context,
                                "Image saved to Pictures/ReReader",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error saving image to gallery", e)
            }
        }

        /**
         * Fixes image orientation based on EXIF data.
         */
        fun fixImageOrientation(context: Context, uri: Uri, bitmap: Bitmap): Bitmap {
            try {
                val inputStream = context.contentResolver.openInputStream(uri)
                val exif = inputStream?.use { ExifInterface(it) }

                val orientation = exif?.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                ) ?: ExifInterface.ORIENTATION_NORMAL

                if (orientation == ExifInterface.ORIENTATION_NORMAL) return bitmap

                val matrix = Matrix()
                when (orientation) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                    ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                    ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                    ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
                    ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
                    else -> return bitmap
                }

                Log.d(TAG, "Rotating image. Orientation tag: $orientation")
                return Bitmap.createBitmap(
                    bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                )

            } catch (e: Exception) {
                Log.e(TAG, "Error reading EXIF orientation", e)
                return bitmap
            }
        }

        // In ImageUtils.kt
        fun bitmapToMat(bitmap: Bitmap): Mat {
            val mat = Mat()
            Utils.bitmapToMat(bitmap, mat)
            return mat
        }

        fun matToBitmap(mat: Mat): Bitmap {
            val bitmap = createBitmap(mat.cols(), mat.rows())
            Utils.matToBitmap(mat, bitmap)
            return bitmap
        }
    }
}