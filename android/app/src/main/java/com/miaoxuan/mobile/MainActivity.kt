package com.miaoxuan.mobile

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.compose.AsyncImage
import com.miaoxuan.mobile.data.GalleryPhoto
import com.miaoxuan.mobile.model.PhotoCandidate
import com.miaoxuan.mobile.model.PickerSettings
import com.miaoxuan.mobile.model.TaskSummary
import java.text.DateFormat
import java.util.Date

class MainActivity : ComponentActivity() {
    private val viewModel: MiaoXuanViewModel by viewModels()

    private fun hasMediaAccess(): Boolean = when {
        Build.VERSION.SDK_INT >= 34 ->
            ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED) == PackageManager.PERMISSION_GRANTED
        Build.VERSION.SDK_INT >= 33 ->
            ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED
        else -> ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
    }

    private fun mediaPermissions(): Array<String> = when {
        Build.VERSION.SDK_INT >= 34 -> arrayOf(
            Manifest.permission.READ_MEDIA_IMAGES,
            Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED,
        )
        Build.VERSION.SDK_INT >= 33 -> arrayOf(Manifest.permission.READ_MEDIA_IMAGES)
        else -> arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        viewModel.loadGallery(hasMediaAccess())
        setContent {
            MaterialTheme {
                val state by viewModel.state.collectAsStateWithLifecycle()
                val permissionLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestMultiplePermissions(),
                ) { viewModel.loadGallery(hasMediaAccess()) }
                val favoriteLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.StartIntentSenderForResult(),
                ) { result -> viewModel.onFavoriteResult(result.resultCode == Activity.RESULT_OK) }
                val applyFavorites = {
                    try {
                        val request = viewModel.favoriteRequest()
                        favoriteLauncher.launch(IntentSenderRequest.Builder(request.intentSender).build())
                    } catch (error: Exception) {
                        viewModel.reportError(error.message ?: "无法请求系统收藏")
                    }
                }
                // Analysis is followed by the required Android confirmation. Recomposition does
                // not launch it twice because the ViewModel consumes the pending flag first.
                LaunchedEffect(state.favoriteRequestPending) {
                    if (state.favoriteRequestPending) {
                        viewModel.consumeFavoriteRequest()
                        applyFavorites()
                    }
                }
                MiaoXuanApp(
                    state = state,
                    onGrantAccess = { permissionLauncher.launch(mediaPermissions()) },
                    onRefresh = { viewModel.loadGallery(hasMediaAccess()) },
                    onToggleSource = viewModel::toggleSelection,
                    onSelectAll = viewModel::selectAll,
                    onAnalyze = viewModel::analyze,
                    onNewTask = viewModel::clearTask,
                    onSettings = viewModel::updateSettings,
                    onToggleRecommendation = viewModel::toggleRecommendation,
                    onApplyFavorites = applyFavorites,
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MiaoXuanApp(
    state: MobileUiState,
    onGrantAccess: () -> Unit,
    onRefresh: () -> Unit,
    onToggleSource: (android.net.Uri) -> Unit,
    onSelectAll: () -> Unit,
    onAnalyze: () -> Unit,
    onNewTask: () -> Unit,
    onSettings: (PickerSettings) -> Unit,
    onToggleRecommendation: (android.net.Uri) -> Unit,
    onApplyFavorites: () -> Unit,
) {
    var showSettings by remember { mutableStateOf(false) }
    var showHistory by remember { mutableStateOf(false) }
    var selectedPhoto by remember { mutableStateOf<PhotoCandidate?>(null) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Column { Text("秒选", fontWeight = FontWeight.Bold); Text("自动挑选 · 相册收藏", style = MaterialTheme.typography.labelSmall) } },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
                actions = {
                    IconButton(onClick = { showHistory = true }) { Icon(Icons.Default.History, "任务历史") }
                    IconButton(onClick = { showSettings = true }) { Icon(Icons.Default.Settings, "挑图设置") }
                },
            )
        },
    ) { padding ->
        when {
            state.isAnalyzing -> AnalyzingContent(state, Modifier.padding(padding))
            state.photos.isNotEmpty() -> ResultsContent(
                state, Modifier.padding(padding), onNewTask, onApplyFavorites,
            ) { selectedPhoto = it }
            else -> GalleryContent(
                state, Modifier.padding(padding), onGrantAccess, onRefresh,
                onToggleSource, onSelectAll, onAnalyze,
            )
        }
    }
    if (showSettings) PickerSettingsDialog(state.settings, { onSettings(it); showSettings = false }, { showSettings = false })
    if (showHistory) HistorySheet(state.history) { showHistory = false }
    selectedPhoto?.let { photo ->
        ReviewDialog(photo, {
            onToggleRecommendation(photo.uri)
            selectedPhoto = photo.copy(recommended = !photo.recommended)
        }, { selectedPhoto = null })
    }
}

@Composable
private fun GalleryContent(
    state: MobileUiState,
    modifier: Modifier,
    onGrantAccess: () -> Unit,
    onRefresh: () -> Unit,
    onToggleSource: (android.net.Uri) -> Unit,
    onSelectAll: () -> Unit,
    onAnalyze: () -> Unit,
) {
    if (!state.hasMediaAccess) {
        Column(modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(Icons.Default.Favorite, null, modifier = Modifier.size(56.dp), tint = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.height(24.dp))
            Text("从手机相册挑出好照片", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(12.dp))
            Text("授权后可批量选择本机照片。Android 14 及以上也可只授权指定照片。", textAlign = TextAlign.Center)
            Spacer(Modifier.height(24.dp))
            Button(onClick = onGrantAccess, modifier = Modifier.fillMaxWidth()) { Text("允许访问相册") }
        }
        return
    }
    Column(modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("手机相册", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                Text("已选 ${state.selectedUris.size} / ${state.gallery.size} 张", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            TextButton(onClick = onRefresh) { Text("刷新") }
            TextButton(onClick = onSelectAll, enabled = state.gallery.isNotEmpty()) {
                Text(if (state.selectedUris.size == state.gallery.size) "取消全选" else "全选")
            }
        }
        if (state.isLoadingGallery) Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
        else if (state.gallery.isEmpty()) Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            Text("暂无可访问照片。拍照后刷新，或重新授权相册。", modifier = Modifier.padding(24.dp), textAlign = TextAlign.Center)
        }
        else LazyVerticalGrid(
            columns = GridCells.Adaptive(110.dp),
            contentPadding = PaddingValues(8.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
            modifier = Modifier.weight(1f),
        ) {
            items(state.gallery, key = { it.uri }) { photo ->
                GalleryCard(photo, photo.uri in state.selectedUris) { onToggleSource(photo.uri) }
            }
        }
        state.error?.let { Text(it, color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(horizontal = 16.dp)) }
        HorizontalDivider()
        Button(onClick = onAnalyze, enabled = state.selectedUris.isNotEmpty(), modifier = Modifier.fillMaxWidth().padding(12.dp)) {
            Icon(Icons.Default.AutoAwesome, null)
            Spacer(Modifier.width(8.dp))
            Text("分析所选照片并加入相册收藏")
        }
    }
}

@Composable
private fun GalleryCard(photo: GalleryPhoto, selected: Boolean, onClick: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth().clickable(onClick = onClick)) {
        Box {
            AsyncImage(model = photo.uri, contentDescription = photo.displayName, modifier = Modifier.fillMaxWidth().height(116.dp), contentScale = ContentScale.Crop)
            if (selected) Surface(color = MaterialTheme.colorScheme.primary, shape = RoundedCornerShape(bottomStart = 10.dp), modifier = Modifier.align(Alignment.TopEnd)) {
                Icon(Icons.Default.CheckCircle, "已选择", tint = Color.White, modifier = Modifier.padding(4.dp))
            }
            if (photo.isFavorite) Icon(Icons.Default.Favorite, "已收藏", tint = Color.Red, modifier = Modifier.align(Alignment.BottomEnd).padding(5.dp))
        }
    }
}

@Composable
private fun AnalyzingContent(state: MobileUiState, modifier: Modifier) {
    Column(modifier.fillMaxSize().padding(32.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        CircularProgressIndicator()
        Spacer(Modifier.height(24.dp))
        Text("正在本机分析照片", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(8.dp))
        Text("${state.analyzedCount} / ${state.selectedUris.size}", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(20.dp))
        Text("完成后会弹出 Android 的收藏确认。", textAlign = TextAlign.Center)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ResultsContent(
    state: MobileUiState,
    modifier: Modifier,
    onNewTask: () -> Unit,
    onApplyFavorites: () -> Unit,
    onPhotoClick: (PhotoCandidate) -> Unit,
) {
    var filter by remember { mutableIntStateOf(-1) }
    val visible = state.photos.filter { filter == -1 || it.recommended == (filter == 1) }
    val recommendedCount = state.photos.count { it.recommended }
    Column(modifier.fillMaxSize()) {
        Column(Modifier.padding(horizontal = 16.dp, vertical = 10.dp)) {
            Text("已推荐 $recommendedCount 张好照片", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            Text("共分析 ${state.photos.size} 张 · 点击照片可调整推荐", color = MaterialTheme.colorScheme.onSurfaceVariant)
            state.message?.let { Text(it, color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(top = 8.dp)) }
            state.error?.let { Text(it, color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 8.dp)) }
            Spacer(Modifier.height(12.dp))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = filter == -1, onClick = { filter = -1 }, label = { Text("全部 ${state.photos.size}") })
                FilterChip(selected = filter == 1, onClick = { filter = 1 }, label = { Text("推荐 $recommendedCount") })
                FilterChip(selected = filter == 0, onClick = { filter = 0 }, label = { Text("未推荐 ${state.photos.size - recommendedCount}") })
            }
        }
        LazyVerticalGrid(
            columns = GridCells.Adaptive(150.dp),
            contentPadding = PaddingValues(12.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            modifier = Modifier.weight(1f),
        ) { items(visible, key = { it.uri }) { PhotoCard(it, onPhotoClick) } }
        HorizontalDivider()
        Row(Modifier.fillMaxWidth().padding(12.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlinedButton(onClick = onNewTask, modifier = Modifier.weight(1f)) { Text("新建任务") }
            Button(onClick = onApplyFavorites, enabled = recommendedCount > 0, modifier = Modifier.weight(1.5f)) {
                Icon(Icons.Default.Favorite, null)
                Spacer(Modifier.width(6.dp))
                Text("加入相册收藏")
            }
        }
    }
}

@Composable
private fun PhotoCard(photo: PhotoCandidate, onClick: (PhotoCandidate) -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth().clickable { onClick(photo) },
        colors = CardDefaults.cardColors(containerColor = if (photo.recommended) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surfaceContainerLow),
    ) {
        Box {
            AsyncImage(model = photo.uri, contentDescription = photo.displayName, modifier = Modifier.fillMaxWidth().height(138.dp), contentScale = ContentScale.Crop)
            Surface(color = Color.Black.copy(alpha = .68f), shape = RoundedCornerShape(bottomStart = 10.dp), modifier = Modifier.align(Alignment.TopEnd)) {
                Text(if (photo.recommended) "推荐" else "未推荐", color = Color.White, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp), fontWeight = FontWeight.Bold)
            }
        }
        Column(Modifier.padding(10.dp)) {
            Text(photo.displayName, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.labelLarge)
            Text("第 ${photo.groupId} 组 · ${photo.score.toInt()} 分", style = MaterialTheme.typography.labelSmall)
            if (photo.reasons.isNotEmpty()) Text(photo.reasons.first(), maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.error)
        }
    }
}

@Composable
private fun ReviewDialog(photo: PhotoCandidate, onToggle: () -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(photo.displayName, maxLines = 1, overflow = TextOverflow.Ellipsis) },
        text = {
            Column {
                AsyncImage(model = photo.uri, contentDescription = photo.displayName, modifier = Modifier.fillMaxWidth().height(260.dp).clip(RoundedCornerShape(16.dp)), contentScale = ContentScale.Fit)
                Spacer(Modifier.height(12.dp))
                Text("第 ${photo.groupId} 组 · 综合 ${photo.score.toInt()} 分")
                photo.reasons.forEach { Text(it, color = MaterialTheme.colorScheme.error) }
                Spacer(Modifier.height(10.dp))
                OutlinedButton(onClick = onToggle, modifier = Modifier.fillMaxWidth()) {
                    Text(if (photo.recommended) "取消推荐" else "设为推荐")
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("完成") } },
    )
}

@Composable
private fun PickerSettingsDialog(current: PickerSettings, onSave: (PickerSettings) -> Unit, onDismiss: () -> Unit) {
    var keepMax by remember { mutableIntStateOf(current.maximumPerBurst) }
    var target by remember { mutableStateOf(current.targetKeepCount?.toString().orEmpty()) }
    var blur by remember { mutableStateOf(current.detectBlur) }
    var exposure by remember { mutableStateOf(current.detectExposure) }
    var duplicates by remember { mutableStateOf(current.detectDuplicates) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("挑图设置") },
        text = {
            Column {
                Text("每组最多推荐 $keepMax 张", fontWeight = FontWeight.Medium)
                Slider(value = keepMax.toFloat(), onValueChange = { keepMax = it.toInt().coerceIn(1, 5) }, valueRange = 1f..5f, steps = 3)
                OutlinedTextField(value = target, onValueChange = { target = it.filter(Char::isDigit) }, label = { Text("目标推荐张数（可选）") }, singleLine = true)
                ToggleRow("检测明显虚焦", blur) { blur = it }
                ToggleRow("检测严重曝光异常", exposure) { exposure = it }
                ToggleRow("识别近重复连拍", duplicates) { duplicates = it }
            }
        },
        confirmButton = { TextButton(onClick = { onSave(PickerSettings(maximumPerBurst = keepMax, targetKeepCount = target.toIntOrNull(), detectBlur = blur, detectExposure = exposure, detectDuplicates = duplicates)) }) { Text("保存") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消") } },
    )
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, modifier = Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun HistorySheet(history: List<TaskSummary>, onDismiss: () -> Unit) {
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = rememberModalBottomSheetState()) {
        Text("任务历史", modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        if (history.isEmpty()) Text("还没有完成的筛选任务。", modifier = Modifier.padding(24.dp))
        else LazyColumn(contentPadding = PaddingValues(bottom = 32.dp)) {
            items(history) { task ->
                Row(Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 14.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.AutoAwesome, null, tint = MaterialTheme.colorScheme.primary)
                    Spacer(Modifier.width(14.dp))
                    Column(Modifier.weight(1f)) {
                        Text(task.title, fontWeight = FontWeight.Medium)
                        Text(DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT).format(Date(task.analyzedAtMillis)), style = MaterialTheme.typography.labelSmall)
                    }
                    Text("${task.keepCount}/${task.sourceCount}", fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}
