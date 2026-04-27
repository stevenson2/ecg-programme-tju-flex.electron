# ESP32-ECG 手机客户端 App

## 环境要求

- Flutter SDK >= 3.0.0
- Dart SDK >= 3.0.0
- Android Studio / Xcode

## 首次运行

```bash
# 1. 创建 Flutter 项目脚手架
flutter create --project-name ecg_app .

# 2. 覆盖 lib 目录
#    将本目录的 lib/main.dart 等文件复制到生成的项目中

# 3. 安装依赖
flutter pub get

# 4. 连接手机，运行
flutter run
```

> 注意：`pubspec.yaml` 中的 `flutter_blue_plus` 需要 Android SDK 21+、iOS 13+

## 连接步骤

1. ESP32 上电（确保 BLE 广播 "ESP32-ECG"）
2. 手机打开 App，点击 **「连接 ESP32」**
3. 自动扫描并连接，成功后实时显示波形

## 波形颜色

| 颜色 | 信号 |
|:---:|:---|
| 🟢 绿色 | 纯净心电（无噪声） |
| 🔴 红色 | 原始采集信号（含噪声） |
| 🔵 蓝色 | 滤波后信号 |
| ⚪ 黄色竖线 | 200ms 延迟参考线 |
