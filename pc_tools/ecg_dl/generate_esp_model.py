#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_esp_model.py -- generate ESP-IDF model.cc/model.h for v3-A INT8.

Writes into:
  - experiments/esp_idf_ecg_migration/main
  - C:/esp/esp_idf_ecg_migration/main (build copy), if present

Also updates main.cc to reference the v3-A symbol.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPO_MAIN = REPO / "experiments" / "esp_idf_ecg_migration" / "main"
_esp_win = Path("C:/esp/esp_idf_ecg_migration/main")
_esp_wsl = Path("/mnt/c/esp/esp_idf_ecg_migration/main")
ESP_MAIN = _esp_wsl if _esp_wsl.exists() else _esp_win
TFLITE = Path(__file__).resolve().parent / "models" / "deploy_match" / "ecg_model_v3a_int8.tflite"
SYMBOL = "models_ecg_model_v3a_int8_tflite"


def write_model(dirpath):
    data = TFLITE.read_bytes()
    lines = [f"unsigned char {SYMBOL}[] = {{"]
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hexes = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {hexes},")
    lines.append("};")
    lines.append(f"unsigned int {SYMBOL}_len = {len(data)};")
    lines.append("")
    (dirpath / "model.cc").write_text("\n".join(lines), encoding="utf-8")
    (dirpath / "model.h").write_text(
        "#ifndef MODEL_H\n#define MODEL_H\n"
        f"extern const unsigned char {SYMBOL}[];\n"
        f"extern const unsigned int {SYMBOL}_len;\n"
        "#endif\n", encoding="utf-8")
    # Update main.cc symbol reference if present.
    main = dirpath / "main.cc"
    if main.exists():
        txt = main.read_text(encoding="utf-8")
        txt = txt.replace("models_ecg_model_exp7c_int8_tflite", SYMBOL)
        main.write_text(txt, encoding="utf-8")
    print(f"[GEN] wrote {dirpath / 'model.cc'} ({len(data)} bytes)", flush=True)


def main():
    write_model(REPO_MAIN)
    if ESP_MAIN.exists():
        write_model(ESP_MAIN)
    else:
        print("[GEN] C:/esp build copy not found; only repo updated", flush=True)


if __name__ == "__main__":
    main()
