package com.miaoxuan.mobile

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
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
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.UploadFile
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
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
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.compose.AsyncImage
import com.miaoxuan.mobile.model.PhotoCandidate
import com.miaoxuan.mobile.model.PickerSettings
import com.miaoxuan.mobile.model.TaskSummary
import java.text.DateFormat
import java.util.Date

class MainActivity : ComponentActivity() {
    private val viewModel: MiaoXuanViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                val state by viewModel.state.collectAsStateWithLifecycle()
                val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
                    uris.forEach { uri ->
                        runCatching {
                            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                    }
                    if (uris.isNotEmpty()) viewModel.setSourceUris(uris)
                }
                val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { uri ->
                    if (uri != null) contentResolver.openOutputStream(uri)?.use(viewModel::writeCsv)
                }
                MiaoXuanApp(
                    state = state,
                    onImport = { importLauncher.launch(arrayOf("image/*")) },
                    onAnalyze = viewModel::analyze,
                    onNewTask = viewModel::clearTask,
                    onSettings = viewModel::updateSettings,
                    onRating = viewModel::setRating,
                    onExport = { exportLauncher.launch("miaoxuan-selection.csv") },
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MiaoXuanApp(
    state: MobileUiState,
    onImport: () -> Unit,
    onAnalyze: () -> Unit,
    onNewTask: () -> Unit,
    onSettings: (PickerSettings) -> Unit,
    onRating: (android.net.Uri, Int) -> Unit,
    onExport: () -> Unit,
) {
    var showSettings by remember { mutableStateOf(false) }
    var showHistory by remember { mutableStateOf(false) }
    var selectedPhoto by remember { mutableStateOf<PhotoCandidate?>(null) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Column { Text("秒选", fontWeight = FontWeight.Bold); Text("本机离线智能筛片", style = MaterialTheme.typography.labelSmall) } },
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
            state.photos.isNotEmpty() -> ResultsContent(state, Modifier.padding(padding), onNewTask, onExport) { selectedPhoto = it }
            else -> ImportContent(state, Modifier.padding(padding), onImport, onAnalyze)
        }
    }
    if (showSettings) PickerSettingsDialog(state.settings, { onSettings(it); showSettings = false }, { showSettings = false })
    if (showHistory) HistorySheet(state.history) { showHistory = false }
    selectedPhoto?.let { photo -> ReviewDialog(photo, { onRating(photo.uri, it); selectedPhoto = photo.copy(rating = it) }, { selectedPhoto = null }) }
}

@Composable
private fun ImportContent(state: MobileUiState, modifier: Modifier, onImport: () -> Unit, onAnalyze: () -> Unit) {
    Column(
        modifier = modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Surface(shape = RoundedCornerShape(28.dp), color = MaterialTheme.colorScheme.primaryContainer) {
            Icon(Icons.Default.AutoAwesome, null, modifier = Modifier.padding(28.dp).size(48.dp), tint = MaterialTheme.colorScheme.primary)
        }
        Spacer(Modifier.height(24.dp))
        Text(if (state.sourceUris.isEmpty()) "先导入一组照片" else "已导入 ${state.sourceUris.size} 张照片", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(10.dp))
        Text(
            if (state.sourceUris.isEmpty()) "秒选只读取你选中的照片，在手机本机完成初筛。"
            else "确认挑图规则后开始分析；原图不会被修改。",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(28.dp))
        Button(onClick = onImport, modifier = Modifier.fillMaxWidth()) { Icon(Icons.Default.Image, null); Spacer(Modifier.width(8.dp)); Text(if (state.sourceUris.isEmpty()) "选择照片" else "重新选择照片") }
        if (state.sourceUris.isNotEmpty()) {
            Spacer(Modifier.height(12.dp))
            OutlinedButton(onClick = onAnalyze, modifier = Modifier.fillMaxWidth()) { Icon(Icons.Default.AutoAwesome, null); Spacer(Modifier.width(8.dp)); Text("开始本机分析") }
        }
        state.error?.let { Text(it, color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 16.dp), textAlign = TextAlign.Center) }
        Spacer(Modifier.height(32.dp))
        Text("不会上传照片 · 不删除原图 · 导出 CSV 交接桌面版", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
    }
}

@Composable
private fun AnalyzingContent(state: MobileUiState, modifier: Modifier) {
    Column(modifier.fillMaxSize().padding(32.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        CircularProgressIndicator()
        Spacer(Modifier.height(24.dp))
        Text("正在本机分析照片", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(8.dp))
        Text("${state.analyzedCount} / ${state.sourceUris.size}", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(20.dp))
        Text("正在检查清晰度、曝光和相似连拍；请保持应用在前台。", textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ResultsContent(state: MobileUiState, modifier: Modifier, onNewTask: () -> Unit, onExport: () -> Unit, onPhotoClick: (PhotoCandidate) -> Unit) {
    var filter by remember { mutableIntStateOf(-1) }
    val visible = state.photos.filter { filter == -1 || it.rating == filter }
    val counts = (0..3).associateWith { rating -> state.photos.count { it.rating == rating } }
    Column(modifier.fillMaxSize()) {
        Column(Modifier.padding(horizontal = 16.dp, vertical = 10.dp)) {
            Text("筛选建议已生成", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            Text("${state.photos.size} 张 · 保留 ${state.photos.count { it.rating > 0 }} 张 · 点击照片可人工改星", color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(12.dp))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                FilterChip(selected = filter == -1, onClick = { filter = -1 }, label = { Text("全部 ${state.photos.size}") })
                (3 downTo 0).forEach { rating -> FilterChip(selected = filter == rating, onClick = { filter = rating }, label = { Text("${rating}★ ${counts[rating]}") }) }
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
            Button(onClick = onExport, modifier = Modifier.weight(1f)) { Icon(Icons.Default.UploadFile, null); Spacer(Modifier.width(6.dp)); Text("导出 CSV") }
        }
    }
}

@Composable
private fun PhotoCard(photo: PhotoCandidate, onClick: (PhotoCandidate) -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth().clickable { onClick(photo) },
        colors = CardDefaults.cardColors(containerColor = if (photo.rating == 3) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surfaceContainerLow),
    ) {
        Box {
            AsyncImage(model = photo.uri, contentDescription = photo.displayName, modifier = Modifier.fillMaxWidth().height(138.dp), contentScale = ContentScale.Crop)
            Surface(color = Color.Black.copy(alpha = .68f), shape = RoundedCornerShape(bottomStart = 10.dp), modifier = Modifier.align(Alignment.TopEnd)) {
                Text(if (photo.rating == 0) "未选" else "${photo.rating}★", color = Color.White, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp), fontWeight = FontWeight.Bold)
            }
        }
        Column(Modifier.padding(10.dp)) {
            Text(photo.displayName, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.labelLarge)
            Text("第 ${photo.groupId} 组 · ${photo.score.toInt()} 分", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (photo.reasons.isNotEmpty()) Text(photo.reasons.first(), maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.error)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ReviewDialog(photo: PhotoCandidate, onRating: (Int) -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(photo.displayName, maxLines = 1, overflow = TextOverflow.Ellipsis) },
        text = {
            Column {
                AsyncImage(model = photo.uri, contentDescription = photo.displayName, modifier = Modifier.fillMaxWidth().height(260.dp).clip(RoundedCornerShape(16.dp)), contentScale = ContentScale.Fit)
                Spacer(Modifier.height(12.dp))
                Text("第 ${photo.groupId} 组 · 清晰度 ${photo.sharpness.toInt()} · 综合 ${photo.score.toInt()} 分", style = MaterialTheme.typography.bodySmall)
                if (photo.reasons.isNotEmpty()) FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) { photo.reasons.forEach { AssistChip(onClick = {}, label = { Text(it) }) } }
                Spacer(Modifier.height(8.dp))
                Text("人工星级", style = MaterialTheme.typography.labelLarge)
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) { (0..3).forEach { rating -> FilterChip(selected = photo.rating == rating, onClick = { onRating(rating) }, label = { Text(if (rating == 0) "不选" else "${rating}★") }) } }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Icon(Icons.Default.CheckCircle, null); Spacer(Modifier.width(4.dp)); Text("完成") } },
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
                Text("每组最多保留 $keepMax 张", fontWeight = FontWeight.Medium)
                Slider(value = keepMax.toFloat(), onValueChange = { keepMax = it.toInt().coerceIn(1, 5) }, valueRange = 1f..5f, steps = 3)
                OutlinedTextField(value = target, onValueChange = { target = it.filter(Char::isDigit) }, label = { Text("目标保留张数（可选）") }, singleLine = true)
                ToggleRow("检测明显虚焦", blur) { blur = it }
                ToggleRow("检测严重曝光异常", exposure) { exposure = it }
                ToggleRow("识别近重复连拍", duplicates) { duplicates = it }
                Text("评分只作初筛建议，最终以你的人工复核为准。", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
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
        if (history.isEmpty()) Text("还没有完成的筛选任务。", modifier = Modifier.padding(24.dp), color = MaterialTheme.colorScheme.onSurfaceVariant)
        else LazyColumn(contentPadding = PaddingValues(bottom = 32.dp)) {
            items(history) { task ->
                Row(Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 14.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.AutoAwesome, null, tint = MaterialTheme.colorScheme.primary)
                    Spacer(Modifier.width(14.dp))
                    Column(Modifier.weight(1f)) { Text(task.title, fontWeight = FontWeight.Medium); Text(DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT).format(Date(task.analyzedAtMillis)), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                    Text("${task.keepCount}/${task.sourceCount}", fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}
