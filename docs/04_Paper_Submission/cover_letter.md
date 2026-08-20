# Cover Letter

**To:** The Editors  
*Sensors* (MDPI)  

**Date:** [Date of submission — to be filled]  

**Re:** Submission of manuscript "A Flexible Dry-Electrode ECG System with Dual-Expert On-Device Deep Learning on an ESP32-S3 Microcontroller: System Design, Deployment-Chain-Matched Training, and Patient-Level Evaluation"

---

Dear Editors,

We are pleased to submit the above-titled original research article for consideration for publication in *Sensors* (MDPI). The work presents a complete, low-cost flexible ECG monitoring system with on-device dual-expert deep learning, built around a commercial ESP32-S3 microcontroller, and reports patient-level evaluation across three public databases. We believe the manuscript aligns closely with *Sensors*' scope in wearable sensors, biomedical signal processing, and AI-enabled sensing systems.

**Four Innovation Pillars**

1. **Dual-expert on-device AI fusion.** Two INT8-quantized 1D-CNN experts — one for arrhythmia beat classification (trained on MIT-BIH + INCART) and one for myocardial infarction detection (trained on PTB) — are fused by a logical OR rule on the ESP32-S3, enabling a single-lead wearable to flag both abnormality classes in real time. To our knowledge, arrhythmia-plus-MI dual-expert fusion has not been previously reported on microcontroller-class hardware.

2. **Deployment-chain-matched training.** We quantify the accuracy cost of the mismatch between offline zero-phase (`filtfilt`) preprocessing used in training and the causal biquad + comb filter chain deployed in firmware (up to ΔAUC −0.105 on the PTB domain), and recover approximately half (≈49%) of that cost by retraining on data preprocessed through an exact software replica of the deployed chain. This train-versus-deploy filter mismatch has not been previously quantified for on-device ECG inference on MCU-class hardware.

3. **Patient-level multi-dataset evaluation with leakage audit.** All headline results use patient-level 60/20/20 splits across MIT-BIH, INCART, and PTB. We conduct an explicit leakage audit, isolate historical leakage-affected metrics in a dedicated methodological-contrast section, and report per-class behavior under the AAMI EC57 superclass grouping — exposing per-class ceilings (e.g., fusion beat recall 0.442) that aggregate accuracy hides.

4. **Complete, reproducible, low-cost stack.** The system integrates a self-made PEDOT:PSS:PVA:TA flexible dry electrode (reproducing the formulation of Cao et al., ACS AMI 2022), an AD8232 analog front end, dual-core firmware (digital filtering, Pan-Tompkins heart-rate detection, and TFLite Micro inference under FreeRTOS), and a BLE link to a Flutter smartphone application — all assembled from commercial off-the-shelf parts at a cost of a few tens of US dollars.

**Why Sensors**

*Sensors* is a natural venue for this manuscript. The journal's scope encompasses wearable sensors, biomedical signal acquisition, and AI-enabled sensing — each a core component of our contribution. The system demonstrates a functioning wearable sensor with integrated intelligence, validated through rigorous patient-level evaluation, and documented with full reproducibility. The manuscript also speaks to the growing intersection of flexible electronics and embedded AI that *Sensors* has actively published in recent years (e.g., Farag et al., *Sensors* 2023, 10.3390/s23031365; Zhou et al., *Electronics* 2025, 10.3390/electronics14132654).

**Declarations**

- This manuscript has not been published previously and is not under consideration elsewhere.
- All authors have approved the manuscript and agree with its submission to *Sensors*.
- The authors declare no conflicts of interest.
- The system is a research prototype and is not intended for clinical diagnosis.
- Ethics approval for the human-subject validation protocol is pending review at Tianjin University and will be reported upon completion; the study will not commence before approval is obtained.

**Suggested Reviewers** *(optional — to be discussed with advisors)*

- [Placeholder: 3–5 reviewer candidates with expertise in wearable ECG, embedded AI, flexible electrodes, and/or patient-level ECG evaluation — to be proposed in consultation with Prof. Yang Hui and Prof. Huang Xian.]

We appreciate your time and consideration, and we look forward to your response.

Respectfully,

[First Author Name] (Corresponding Author)  
[Affiliation: Tianjin University, School of [XXXX]]  
[Email: ________]  

Prof. Yang Hui (Co-Corresponding Author)  
Tianjin University, [School/Department]  
[Email: ________]  

Prof. Huang Xian (Co-Corresponding Author)  
Tianjin University, [School/Department]  
[Email: ________]

---

*Note to authors: Fill bracketed placeholders before submission. Confirm corresponding-author designations and reviewer suggestions with advisors.*
