package com.miaoxuan.mobile

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.miaoxuan.mobile.analysis.LocalPhotoAnalyzer
import com.miaoxuan.mobile.data.GalleryPhoto
import com.miaoxuan.mobile.data.MediaGalleryRepository
import com.miaoxuan.mobile.data.TaskHistoryStore
import com.miaoxuan.mobile.model.PhotoCandidate
import com.miaoxuan.mobile.model.PickerSettings
import com.miaoxuan.mobile.model.TaskSummary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class MobileUiState(
    val gallery: List<GalleryPhoto> = emptyList(),
    val selectedUris: Set<Uri> = emptySet(),
    val photos: List<PhotoCandidate> = emptyList(),
    val settings: PickerSettings = PickerSettings(),
    val isLoadingGallery: Boolean = false,
    val isAnalyzing: Boolean = false,
    val analyzedCount: Int = 0,
    val error: String? = null,
    val message: String? = null,
    val hasMediaAccess: Boolean = false,
    val favoriteRequestPending: Boolean = false,
    val history: List<TaskSummary> = emptyList(),
)

class MiaoXuanViewModel(application: Application) : AndroidViewModel(application) {
    private val analyzer = LocalPhotoAnalyzer(application)
    private val galleryRepository = MediaGalleryRepository(application)
    private val historyStore = TaskHistoryStore(application)
    private val _state = MutableStateFlow(MobileUiState(history = historyStore.load()))
    val state: StateFlow<MobileUiState> = _state.asStateFlow()

    fun loadGallery(hasAccess: Boolean) {
        _state.update { it.copy(hasMediaAccess = hasAccess, isLoadingGallery = hasAccess, error = null) }
        if (!hasAccess) return
        viewModelScope.launch {
            runCatching { withContext(Dispatchers.IO) { galleryRepository.loadPhotos() } }
                .onSuccess { items ->
                    _state.update { current ->
                        current.copy(
                            gallery = items,
                            selectedUris = current.selectedUris.intersect(items.mapTo(mutableSetOf()) { it.uri }),
                            isLoadingGallery = false,
                        )
                    }
                }
                .onFailure { error ->
                    _state.update { it.copy(isLoadingGallery = false, error = error.message ?: "无法读取相册") }
                }
        }
    }

    fun toggleSelection(uri: Uri) = _state.update { current ->
        val selected = current.selectedUris.toMutableSet()
        if (!selected.add(uri)) selected.remove(uri)
        current.copy(selectedUris = selected)
    }

    fun selectAll() = _state.update { current ->
        current.copy(selectedUris = if (current.selectedUris.size == current.gallery.size) emptySet()
            else current.gallery.mapTo(mutableSetOf()) { it.uri })
    }

    fun updateSettings(settings: PickerSettings) = _state.update { it.copy(settings = settings) }

    fun analyze() {
        val snapshot = state.value
        if (snapshot.selectedUris.isEmpty() || snapshot.isAnalyzing) return
        val uris = snapshot.gallery.map { it.uri }.filter { it in snapshot.selectedUris }
        _state.update { it.copy(isAnalyzing = true, error = null, message = null, analyzedCount = 0) }
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.Default) {
                    analyzer.analyze(uris, snapshot.settings) { completed, _ ->
                        _state.update { current -> current.copy(analyzedCount = completed) }
                    }
                }
            }.onSuccess { photos ->
                historyStore.append(
                    TaskSummary(
                        title = "手机筛选 · ${photos.size} 张",
                        analyzedAtMillis = System.currentTimeMillis(),
                        sourceCount = photos.size,
                        keepCount = photos.count { it.recommended },
                    ),
                )
                _state.update {
                    it.copy(
                        photos = photos,
                        isAnalyzing = false,
                        favoriteRequestPending = photos.any { photo -> photo.recommended },
                        history = historyStore.load(),
                    )
                }
            }.onFailure { throwable ->
                _state.update {
                    it.copy(isAnalyzing = false, error = throwable.message ?: "分析失败，请确认相册访问权限。")
                }
            }
        }
    }

    fun toggleRecommendation(uri: Uri) = _state.update { current ->
        current.copy(photos = current.photos.map { photo ->
            if (photo.uri == uri) photo.copy(recommended = !photo.recommended) else photo
        })
    }

    fun consumeFavoriteRequest() {
        _state.update { it.copy(favoriteRequestPending = false) }
    }

    fun onFavoriteResult(approved: Boolean) {
        if (!approved) {
            _state.update { it.copy(message = "系统未授权收藏。可在复核后再次点击“加入相册收藏”。") }
            return
        }
        viewModelScope.launch {
            val recommended = state.value.photos.filter { it.recommended }
            val verified = withContext(Dispatchers.IO) {
                recommended.count { photo -> runCatching { galleryRepository.isFavorite(photo.uri) }.getOrNull() == true }
            }
            _state.update {
                it.copy(message = "已在系统相册收藏 $verified / ${recommended.size} 张推荐照片。")
            }
            loadGallery(true)
        }
    }

    fun reportError(message: String) = _state.update { it.copy(error = message) }

    fun clearTask() = _state.update {
        it.copy(selectedUris = emptySet(), photos = emptyList(), analyzedCount = 0, error = null, message = null)
    }

    fun favoriteRequest() = galleryRepository.favoriteRequest(
        state.value.photos.filter { it.recommended }.map { it.uri }.distinct(),
    )
}
