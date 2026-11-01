# 公开 Icentia11k 触发流复刻（2026-10-31 / 11-01）

在公开 Icentia11k 上复刻板上 62,834 参数网络，比较训练窗定义和几种输入/输出改动。结论与数字见 [RESULTS.md](RESULTS.md)。

## 本目录有什么

- `code/`：训练、评测、剖析脚本，以及固件 AI 链的主机复刻 `hostchain.c`。网络定义沿用 `experiments/nv_expand200_20261026/` 的 `models.py` 与 `source_model.py`。
- `results/`：每次训练的 `train_log.json`、`eval.json`，窗口平移试验的 `shift_test.json`，以及三份汇总：`analysis.json.before_wave1`（配方对照）、`analysis.json`（冒烟测试两波）、`dissect.json`（失败模式剖析）。
- `logs/`：汇总脚本的文本输出。
- `code/split.json`：140 名患者的 train 80 / dev 20 / test 40 划分，抽样种子 20261031，划分种子 7。

## 数据

`data/` 用 Git LFS 存放云端机器上的全部产物，约 2.7 GB：

- `data/cache/`：三段信号流、触发与标签、标量特征。`train_stream.npy` 为 1.3 GB。
- `data/runs/`：每次训练的权重 `epNN.weights.h5` 和测试集逐触发分数 `test_trigger_margins_epNN.npy`。
- `data/runs_parity/trig_s42/`：基线复现。

拉取：

```powershell
git lfs install
git lfs pull
```

没有执行 `git lfs pull` 时，这些文件只有指针，不是数据。

复跑需要：Icentia11k 公开文件、`code/hostchain.c` 编译出的 `libhostchain.so`、TensorFlow 2.21、NumPy、SciPy、wfdb。`train.py` 是第一轮配方，`train2.py` 是冒烟测试的各个臂。
