package com.miaoxuan.mobile.data

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import android.provider.MediaStore.Images.Media

data class GalleryPhoto(
    val uri: Uri,
    val displayName: String,
    val capturedAtMillis: Long,
    val isFavorite: Boolean,
)

/** One owner for local gallery reads and system favorite state. */
class MediaGalleryRepository(context: Context) {
    private val resolver: ContentResolver = context.contentResolver

    fun loadPhotos(limit: Int = 2_000): List<GalleryPhoto> {
        val projection = arrayOf(
            Media._ID,
            Media.DISPLAY_NAME,
            Media.DATE_TAKEN,
            Media.DATE_ADDED,
            Media.VOLUME_NAME,
            Media.IS_FAVORITE,
        )
        val result = ArrayList<GalleryPhoto>()
        resolver.query(
            Media.EXTERNAL_CONTENT_URI,
            projection,
            null,
            null,
            "${Media.DATE_ADDED} DESC",
        )?.use { cursor ->
            val idColumn = cursor.getColumnIndexOrThrow(Media._ID)
            val nameColumn = cursor.getColumnIndexOrThrow(Media.DISPLAY_NAME)
            val takenColumn = cursor.getColumnIndexOrThrow(Media.DATE_TAKEN)
            val addedColumn = cursor.getColumnIndexOrThrow(Media.DATE_ADDED)
            val volumeColumn = cursor.getColumnIndexOrThrow(Media.VOLUME_NAME)
            val favoriteColumn = cursor.getColumnIndexOrThrow(Media.IS_FAVORITE)
            while (cursor.moveToNext() && result.size < limit) {
                val id = cursor.getLong(idColumn)
                val volume = cursor.getString(volumeColumn) ?: MediaStore.VOLUME_EXTERNAL_PRIMARY
                result += GalleryPhoto(
                    uri = Media.getContentUri(volume, id),
                    displayName = cursor.getString(nameColumn) ?: "照片 $id",
                    capturedAtMillis = cursor.getLong(takenColumn).takeIf { it > 0 }
                        ?: cursor.getLong(addedColumn) * 1_000L,
                    isFavorite = cursor.getInt(favoriteColumn) == 1,
                )
            }
        }
        return result
    }

    fun isFavorite(uri: Uri): Boolean? = resolver.query(
        uri,
        arrayOf(Media.IS_FAVORITE),
        null,
        null,
        null,
    )?.use { cursor -> if (cursor.moveToFirst()) cursor.getInt(0) == 1 else null }

    /** The platform presents one confirmation dialog and applies the change before returning. */
    fun favoriteRequest(uris: List<Uri>) = MediaStore.createFavoriteRequest(resolver, uris, true)
}
