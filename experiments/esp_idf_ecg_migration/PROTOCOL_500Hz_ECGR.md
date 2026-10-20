# 500 Hz / 忠实落盘重录协议（2026-10-20）

供 Gallium_Cai 真机重录，替换 20261019 仅有串口 ~100 Hz WAVE 的会话。

## 背景（必须读）

| 通道 | 实际速率 | 抗混叠 | 能否当金标准 |
|------|----------|--------|--------------|
| ADC / AFE | **500 Hz** | — | 源头 |
| 板上 AI / 心率链 | 500→**250 Hz**（2:1，前有 LP≈40 Hz） | 有 | **可对齐学生契约** |
| ECGR（`REC_START`） | **250 Hz** int16 | 同分析链前级 | **推荐主归档** |
| 串口 `WAVE 1` | ~**100 Hz**（frame%5） | **无**专门 AA | **不能**当金标准（易混叠细抖） |

结论：产品 AI 契约是 250@250，**ECGR 250 Hz 已够重训/误报复盘**。若要单独证明「抖是不是 500→100 混叠」，再加短时 500 Hz 调试导出（见下）。

## A. 标准重录（推荐，每次必做）

**环境**
- 标准 Lead I：右臂(−) / 左臂(+) / 右腿参考（AD8232 惯用）
- 另录可穿戴胸贴方位各 ≥1 段
- 静坐 5 min → 慢走 5 min → 坐下恢复 2 min（同一文件连续或分段均可）

**串口命令顺序**
```
WAVE 0
REC_START
# …佩戴活动…
REC_STOP
REC_LIST
REC_STATUS
```

**验收**
- `REC_LIST` 非 empty；`post-list` 有记录
- 用 PC 工具拉出 `.ecgr`，确认 `sample_rate==250`，时长 ≥ 600 s
- 文件放到：`real_captures/inbox/ecgr_YYYYMMDD_HHMMSS_leadI_*.ecgr`
- 元数据旁路记：导联、电极、活动阶段、固件版本、是否开 AI 报警

**禁止**
- 只开 `WAVE 1` 当归档
- 未 `REC_STOP` 就拔电（头部可能 totalSamples=0）

## B. 调试：需要看 500 Hz 原始抖时

当前固件 ECGR 固定 250。短时诊断用其一：

1. **逻辑分析仪 / 板载调试缓冲**（若已开）：抓 AFE 500 Hz 原始 30–60 s  
2. **临时固件**（未默认合并）：将 `WAVE` 改为每帧打印（500 Hz）仅短录，USB 可能丢字节——只做 bench，不做长程  
3. **后续增强（待办）**：`REC_RATE 500` 写双倍 ECGR 或并行 RAW 分区（SPIFFS 空间需评估）

对本轮「伪影 vs 错判」：优先把 **≥30 min Lead I ECGR@250** 录齐，比纠结 500 更重要。

## C. 命名与入库

```
ecgr_<date>_<time>_leadI_standard_sitwalk.ecgr
ecgr_<date>_<time>_wearable_Lchest_sitwalk.ecgr
meta_<same>.json   # fw, electrode, protocol, notes
```

入库后跑：`benchmark_leadI_20261019/scripts/02b_real_captures.py`（或后续 abstain 脚本）更新真机误报表。

## D. 与报警新规则

重录回放请用：**8/10 确认、hold 12 s、解除 5 窗≤1、冷却 25 s**。报告 **进入次数** 与 **报警秒**，不要只报外推 FP/h（短会话会饱和）。
