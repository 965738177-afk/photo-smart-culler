# 秒选 Android

原生 Kotlin + Jetpack Compose 手机版，面向手机本地相册的**离线批量挑图**。

## 已实现

- 使用 Android 媒体库显示本地相册，支持多选与全选；Android 14+ 可以只授权指定照片。
- 本机清晰度、曝光和近重复检测；按连拍时间/画面相似度分组，选出组内最佳照片。
- 分析完成后请求系统批量加入“收藏/喜欢”；也可以先逐张调整推荐再重新应用。
- 在写入请求返回后回读 Android `MediaStore.IS_FAVORITE`，显示成功数量。
- 挑图设置、当前任务统计、任务历史。

## 有意保留的边界

Android 11（API 30）以上才支持系统批量收藏接口。每次设置收藏都需要用户在 Android 弹窗中批准；应用不会改写图像像素或删除照片。此接口设置的是设备本地媒体库的收藏状态。部分品牌相册或云相册有自己的“喜欢”数据，是否同步显示需在目标机型实测。

人脸表情、RAW 内嵌预览与偏好校准尚未迁移到端侧模型，不能把本版评分当成最终选片结论。

## 构建

```bash
cd android
./gradlew testDebugUnitTest assembleDebug
```

APK 输出：`app/build/outputs/apk/debug/app-debug.apk`。
