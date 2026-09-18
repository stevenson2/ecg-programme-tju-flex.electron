#!/usr/bin/env python3
"""P2-1 令牌迁移：把 App 内联视觉魔法数字替换为 app_theme.dart 令牌引用。

用法:
    python tool/migrate_tokens.py --dry-run    # 预览
    python tool/migrate_tokens.py --apply      # 执行

纪律：本脚本本身入库，替换规则可审计；改完必须 flutter analyze 零新增告警。
"""
import argparse
import re
from pathlib import Path

LIB = Path(__file__).resolve().parent.parent / "lib"

# 颜色映射（值 -> 令牌名）
COLOR_MAP = {
    "0xFF0D0D1A": "AppColors.background",
    "0xFF1A1A2E": "AppColors.surface",
    "0xFF2A2A3E": "AppColors.surfaceVariant",
    "0xFF0A0A0E": "AppColors.traceBackground",
    "0xFF1A1A30": "AppColors.gridMinor",
    "0xFF334466": "AppColors.traceBaseline",
    "0xFF445566": "AppColors.traceLabel",
    "0xFF00E5FF": "AppColors.traceLine",
    "0xFF00BFFF": "AppColors.primary",
    "0xFFFF5252": "AppColors.alert",
    "0xFFE53935": "AppColors.error",
    "0xFF4CAF50": "AppColors.success",
}

# 字号映射（值 -> 令牌名）
FONT_MAP = {
    "9": "AppFontSize.xxs",
    "11": "AppFontSize.xs",
    "12": "AppFontSize.sm",
    "13": "AppFontSize.md",
    "14": "AppFontSize.lg",
    "15": "AppFontSize.xl",
    "16": "AppFontSize.xxl",
    "18": "AppFontSize.xxxl",
    "20": "AppFontSize.huge",
    "28": "AppFontSize.display",
}

SKIP = {"app_theme.dart"}


def migrate(text: str) -> tuple[str, int]:
    n = 0
    # const Color(0xFF...) -> 令牌（含 const 前缀一并去掉，因为令牌已是 const）
    for val, token in COLOR_MAP.items():
        pat = re.compile(r"(?:const\s+)?Color\(" + val + r"\)")
        text, k = pat.subn(token, text)
        n += k
    # fontSize: <num> -> fontSize: 令牌
    for val, token in FONT_MAP.items():
        pat = re.compile(r"fontSize:\s*" + val + r"(?![0-9.])")
        text, k = pat.subn("fontSize: " + token, text)
        n += k
    return text, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (args.apply or args.dry_run):
        ap.error("need --apply or --dry-run")

    total = 0
    for f in sorted(LIB.rglob("*.dart")):
        if f.name in SKIP:
            continue
        src = f.read_text(encoding="utf-8")
        out, n = migrate(src)
        if n:
            total += n
            print(f"{'APPLY' if args.apply else 'DRY  '} {f.relative_to(LIB.parent)}: {n} 处")
            if args.apply:
                f.write_text(out, encoding="utf-8")
    print(f"--- 合计 {total} 处")


if __name__ == "__main__":
    main()