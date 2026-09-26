package com.miaoxuan.mobile

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.miaoxuan.mobile.analysis.LocalPhotoAnalyzer
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
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class MobileUiState(
    val sourceUris: List<Uri> = emptyList(),
    val photos: List<PhotoCandidate> = emptyList(),
    val settings: PickerSettings = PickerSettings(),
    val isAnalyzing: Boolean = false,
    val analyzedCount: Int = 0,
    val error: String? = null,
    val history: List<TaskSummary> = emptyList(),
)

class MiaoXuanViewModel(application: Application) : AndroidViewModel(application) {
    private val analyzer = LocalPhotoAnalyzer(application)
    private val historyStore = TaskHistoryStore(application)
    private val _state = MutableStateFlow(MobileUiState(history = historyStore.load()))
    val state: StateFlow<MobileUiState> = _state.asStateFlow()

    fun setSourceUris(uris: List<Uri>) {
        _state.update { it.copy(sourceUris = uris.distinct(), photos = emptyList(), error = null, analyzedCount = 0) }
    }

    fun updateSettings(settings: PickerSettings) = _state.update { it.copy(settings = settings) }

    fun analyze() {
        val snapshot = state.value
        if (snapshot.sourceUris.isEmpty() || snapshot.isAnalyzing) return
        _state.update { it.copy(isAnalyzing = true, error = null, analyzedCount = 0) }
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.Default) {
                    analyzer.analyze(snapshot.sourceUris, snapshot.settings) { completed, _ ->
                        _state.update { current -> current.copy(analyzedCount = completed) }
                    }
                }
            }.onSuccess { photos ->
                val summary = TaskSummary(
                    title = "手机筛选 · ${photos.size} 张",
                    analyzedAtMillis = System.currentTimeMillis(),
                    sourceCount = photos.size,
                    keepCount = photos.count { it.rating > 0 },
                )
                historyStore.append(summary)
                _state.update { it.copy(photos = photos, isAnalyzing = false, history = historyStore.load()) }
            }.onFailure { throwable ->
                _state.update {
                    it.copy(isAnalyzing = false, error = throwable.message ?: "分析失败，请确认已授予图片读取权限。")
                }
            }
        }
    }

    fun setRating(uri: Uri, rating: Int) {
        require(rating in 0..3)
        _state.update { current ->
            current.copy(photos = current.photos.map { photo -> if (photo.uri == uri) photo.copy(rating = rating) else photo })
        }
    }

    fun clearTask() = _state.update {
        it.copy(sourceUris = emptyList(), photos = emptyList(), analyzedCount = 0, error = null)
    }

    fun writeCsv(output: OutputStream) {
        val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.CHINA)
        output.bufferedWriter(Charsets.UTF_8).use { writer ->
            writer.appendLine("文件名,内容URI,星级,推荐星级,分组,评分,拍摄时间,原因")
            state.value.photos.forEach { photo ->
                val cells = listOf(
                    photo.displayName,
                    photo.uri.toString(),
                    photo.rating.toString(),
                    photo.suggestedRating.toString(),
                    photo.groupId.toString(),
                    "%.1f".format(Locale.US, photo.score),
                    formatter.format(Date(photo.capturedAtMillis)),
                    photo.reasons.joinToString("；"),
                )
                writer.appendLine(cells.joinToString(",") { csvCell(it) })
            }
        }
    }

    private fun csvCell(value: String): String = "\"${value.replace("\"", "\"\"")}\""
}

