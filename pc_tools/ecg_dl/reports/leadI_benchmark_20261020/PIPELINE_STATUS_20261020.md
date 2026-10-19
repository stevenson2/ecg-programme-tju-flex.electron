# Pipeline status — Lead I student / distill / public replay

- Completed (Asia/Shanghai): 2026-10-20 ~00:50
- Interrupted prior executor resumed; all six steps finished.

| Step | Status | Artifact |
|------|--------|----------|
| 1 Public-DB alarm replay | DONE (full INCART frozen 11 + PTB Lead I normals; FM R-peak only, blind skipped) | `results/PUBLIC_DB_ALARM_REPLAY.md` `public_db_alarm_replay.json` `figures/public_db_alarm_replay_incart.png` |
| 2 Train student | DONE | `models/student_leadI_resnet_large.h5` meta AUROC_test=0.923 |
| 3 Distill FM center | DONE (CPU emb 8k/4k then TF train) | `models/student_leadI_distill_fm_center.h5` AUROC_test=0.914 |
| 4 Full benchmark eval | DONE | `results/STUDENT_BENCHMARK_EVAL.md` `figures/student_benchmark_comparison.png` |
| 5 Git organize | see repo commit | alarm macros in `main.cc` + docs |
| 6 Free exploration SWOT | DONE | `docs/SWOT_LeadI_学生与竞品_20261020.md` |

Honest notes:
- PyTorch in WSL had `cuda=False`; FM embedding for distill ran on CPU (~15 min) then soft targets cached.
- exp5 high scores on frozen test may leak; do not treat as ceiling.
- Distill raised Se, hurt +P/F1 vs hard-label student.