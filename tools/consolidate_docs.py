#!/usr/bin/env python3
"""Clean up documentation: consolidate 4 docs -> 2 docs."""
import re
from pathlib import Path

ROOT = Path(r"C:\Users\cai\OneDrive\Desktop\ecg-programme-tju-flex.electron-master")

# ============================================================
# 1. Fix README.md: replace old AI section with concise summary
# ============================================================
readme = ROOT / "README.md"
text = readme.read_text(encoding="utf-8-sig")

# Find the AI section boundaries
start_marker = "## AI 异常检测概述"
end_marker = "## PC 端工具"

idx_start = text.find(start_marker)
idx_end = text.find(end_marker, idx_start)

if idx_start > 0 and idx_end > idx_start:
    new_ai_section = """## AI 异常检测

### Phase 1 最佳模型

| 指标 | 数值 |
|------|------|
| 数据 | MIT-BIH (87K) + INCART (176K) = 263K beat级心拍 |
| 模型 | CNN v2, 15K 参数, INT8 ~15KB |
| Loss | FocalLoss (γ=1.0, α=0.75, bug已修复) |
| Acc | **93.98%** |
| AUC | **0.9716** |
| Abnormal Recall | 72% |
| 部署 | `models/ecg_model.tflite` (24.8 KB) |

Phase 2 计划: 自监督预训练 + 600K 模型, 目标 Recall ≥88%.
详见 [ROADMAP.md](ROADMAP.md)

---

"""
    text = text[:idx_start] + new_ai_section + text[idx_end:]
    readme.write_text(text, encoding="utf-8-sig")
    print("[OK] README.md updated")
else:
    print(f"[WARN] README markers not found: start={idx_start}, end={idx_end}")

# ============================================================
# 2. Remove duplicates from README (if any leftover from bad edits)
# ============================================================
# Remove orphaned "### 整体流程" that appears after "## PC 端工具"
text = readme.read_text(encoding="utf-8-sig")
# Remove the orphaned diagram block that got duplicated
orphan = "\n### 整体流程\n\n```\nPC (离线训练)"
idx = text.find(orphan)
if idx > 0:
    # Find the end of this orphan block (next ## section)
    next_section = text.find("\n## ", idx + 10)
    if next_section > idx:
        text = text[:idx] + text[next_section:]
        readme.write_text(text, encoding="utf-8-sig")
        print("[OK] Removed orphan section from README")

# ============================================================
# 3. Archive old docs by moving content note
# ============================================================
for old_file in ["ModelPlan.md", "PLAN.md"]:
    fp = ROOT / old_file
    if fp.exists():
        note = f"# ⚠️ ARCHIVED\n\nThis document has been merged into [README.md](README.md) and [ROADMAP.md](ROADMAP.md).\n"
        fp.write_text(note, encoding="utf-8")
        print(f"[OK] Archived {old_file}")

# ============================================================
# 4. Verify ROADMAP.md exists
# ============================================================
roadmap = ROOT / "ROADMAP.md"
if roadmap.exists():
    print(f"[OK] ROADMAP.md exists ({roadmap.stat().st_size} bytes)")
else:
    print("[WARN] ROADMAP.md not found")

print("\n[DONE] Documentation consolidated to 2 files: README.md + ROADMAP.md")
