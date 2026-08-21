#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlatformIO 构建参数注入脚本。

用法：
    $env:ECG_WIFI_AP_PASSWORD="your-strong-password" ; pio run

说明：
    - 若设置了环境变量 ECG_WIFI_AP_PASSWORD，则将其作为字符串宏注入固件，
      覆盖 include/wifi/ecg_wifi.h 中的开发默认值。
    - 未设置时保持默认开发密码，便于本地快速验证。
    - 生产构建请务必通过 CI/部署平台注入强密码，不要把真实密码写进仓库。

其他可注入项后续可继续在此扩展。
"""

import os

Import("env")  # noqa: F821  (PlatformIO 提供的 SCons 环境)


def _inject_string_macro(name: str, value: str) -> None:
    """向固件编译命令追加字符串宏定义：-D NAME="value" """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    env.Append(CPPDEFINES=[(name, '"{}"'.format(escaped))])
    print("[build_flags] {} injected from environment".format(name))


def main() -> None:
    ap_password = os.environ.get("ECG_WIFI_AP_PASSWORD")
    if ap_password:
        _inject_string_macro("ECG_WIFI_AP_PASSWORD", ap_password)


main()
