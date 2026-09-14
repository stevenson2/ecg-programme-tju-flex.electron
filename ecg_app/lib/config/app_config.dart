/// 应用构建期配置。
///
/// 所有可能随部署环境变化的值都从这里读取，通过 `--dart-define` 注入：
///
/// ```bash
/// flutter run --dart-define=CLOUD_BASE_URL=https://api.example.com/v1 \
///             --dart-define=CLOUD_TOKEN=real-token \
///             --dart-define=ECG_AP_PASSWORD=your-strong-password
/// ```
///
/// 未注入时使用本地开发默认值，便于 mock/联调。
class AppConfig {
  AppConfig._();

  /// 云端服务地址。
  static const String cloudBaseUrl = String.fromEnvironment(
    'CLOUD_BASE_URL',
    defaultValue: 'http://127.0.0.1:8000/v1',
  );

  /// 云端接口鉴权 token。
  static const String cloudToken = String.fromEnvironment(
    'CLOUD_TOKEN',
    defaultValue: 'dev-token',
  );

  /// ESP32 热点密码，用于 App 内“连接指南”文案。
  /// 需与固件构建时注入的 ECG_WIFI_AP_PASSWORD 保持一致。
  static const String apPassword = String.fromEnvironment(
    'ECG_AP_PASSWORD',
    defaultValue: '12345678',
  );

  /// ESP32 记录下载服务地址（热点 AP 默认网关）。
  /// P0-4：不再在 RecordApi 内硬编码，统一经 --dart-define=DEVICE_BASE_URL 注入。
  static const String deviceBaseUrl = String.fromEnvironment(
    'DEVICE_BASE_URL',
    defaultValue: 'http://192.168.4.1',
  );

  /// 构建期固件版本兜底（优先使用 STATUS/HELLO 从设备读取的真实值）。
  static const String firmwareVersion = String.fromEnvironment(
    'FIRMWARE_VERSION',
    defaultValue: 'unknown',
  );

  /// 构建期模型标识兜底（优先使用 STATUS 从设备读取的真实值）。
  static const String modelName = String.fromEnvironment(
    'MODEL_NAME',
    defaultValue: 'unknown',
  );
}
