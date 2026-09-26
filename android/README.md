# 秒选 Android

原生 Kotlin + Jetpack Compose 手机版，面向手机图库的**本机离线初筛**。

## 已实现

- Storage Access Framework 多选导入；不申请“读取全部照片”权限。
- 本机清晰度、曝光和近重复检测；按连拍时间/画面相似度分组并给出 1–3★建议。
- 逐张人工复核和调星、筛选规则、当前任务统计、任务历史。
- 导出 CSV（文件名、URI、星级、分析理由），方便回到桌面版或 Lightroom 流程继续处理。

## 有意保留的边界

Android 的分区存储不适合在没有逐文件授权和完整备份机制的前提下批量改写 RAW/JPG/XMP。因此首版**不改写原图或 XMP**，也不做删除操作；导出的 CSV 是明确、可回滚的交接格式。桌面版仍负责 Lightroom/XMP 写回。

人脸表情、RAW 内嵌预览与偏好校准尚未迁移到端侧模型，不能把本版评分当成最终选片结论。

## 构建

```bash
cd android
./gradlew testDebugUnitTest assembleDebug
```

APK 输出：`app/build/outputs/apk/debug/app-debug.apk`。

