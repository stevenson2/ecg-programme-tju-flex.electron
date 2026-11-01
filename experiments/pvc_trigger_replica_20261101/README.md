# 公开 Icentia11k 触发流复刻（2026-10-31 / 11-01）

在公开 Icentia11k 上复刻板上 62,834 参数网络，比较训练窗定义和几种输入/输出改动。结论与数字见 [RESULTS.md](RESULTS.md)。

## 本目录有什么

- `code/`：训练、评测、剖析脚本，以及固件 AI 链的主机复刻 `hostchain.c`。网络定义沿用 `experiments/nv_expand200_20261026/` 的 `models.py` 与 `source_model.py`。
- `results/`：每次训练的 `train_log.json`、`eval.json`，窗口平移试验的 `shift_test.json`，以及三份汇总：`analysis.json.before_wave1`（配方对照）、`analysis.json`（冒烟测试两波）、`dissect.json`（失败模式剖析）。
- `logs/`：汇总脚本的文本输出。
- `code/split.json`：140 名患者的 train 80 / dev 20 / test 40 划分，抽样种子 20261031，划分种子 7。

## 本目录没有什么

按仓库规则，没有上传原始或派生波形、逐拍标签、逐触发分数、模型权重和日志。这些留在生成它们的云端机器 `/tmp/pvcexp`（`cache/` 约 2.4 GB，`runs/` 约 283 MB）。该目录在临时盘上，机器销毁后不保留。

复跑需要：Icentia11k 公开文件、`code/hostchain.c` 编译出的 `libhostchain.so`、TensorFlow 2.21、NumPy、SciPy、wfdb。`train.py` 是第一轮配方，`train2.py` 是冒烟测试的各个臂。
