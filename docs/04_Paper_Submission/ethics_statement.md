# Ethics Statement — 伦理声明

> **Bilingual: English (primary) + 中文 (supplementary reference).**  
> **Status: DRAFT — ethics approval is PENDING. All claims must be verified with the supervisory team before submission.**

---

## English — Institutional Review and Informed Consent

### Human Subjects Research

The human-subject validation protocol described in this manuscript (Section 4.5) involves healthy adult volunteers wearing a self-made PEDOT:PSS flexible dry electrode and a commercial Ag/AgCl gel electrode for ECG signal quality comparison. The protocol is a device prototype validation, not a clinical trial, and the system is a **research prototype not intended for clinical diagnosis**.

### Ethics Approval Status

**Ethics approval is pending at the time of manuscript preparation.** The study protocol, informed consent form, and supporting materials have been prepared in accordance with the Declaration of Helsinki and relevant institutional guidelines at Tianjin University (TJU). The study will be formally submitted for review through the TJU simplified ethics review path (or advisor-filed internal record, as confirmed with the supervisory team). The approval identifier and review date will be inserted into this statement and the manuscript upon completion.

**The study will not commence, and no human-subject data will be collected, before ethics approval is obtained.**

### Informed Consent

All prospective participants will receive the full informed consent document (drafted in Chinese, `docs/hardware/consent_form_zh.md`) at least 5 minutes before the experiment. The consent form covers:

- Study purpose and procedures (35 minutes total, six action segments per electrode type)
- Potential risks (mild skin irritation, electrode detachment, minimal exercise fatigue)
- Data anonymization protocol (subject codes S01–S10; only sex and age recorded)
- Data use (aggregated statistics for academic publication; raw anonymized waveforms as supplementary material)
- Right to withdraw unconditionally at any time without consequence
- Contact information for the principal investigator and the TJU academic ethics committee

Only participants who sign the informed consent form after having all questions answered will be enrolled.

### Safety

The acquisition system is USB-powered (5 V, ≤500 mA). Subject isolation is provided by the AD8232 instrumentation amplifier (single 3.3 V supply, CMRR ≥86 dB). Electrode materials (PEDOT:PSS, PVA, tannic acid) are widely reported as biocompatible in the epidermal electronics literature, but transient mild skin irritation cannot be excluded and is covered by study stop rules.

### Declaration

- This research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.
- The system described is a research prototype; it is not a medical device and is not CE-marked or FDA-cleared.

---

## 中文 — 机构审查与知情同意

### 人体实验

本文所述人体验证方案（第 4.5 节）涉及健康成年志愿者佩戴自制 PEDOT:PSS 柔性干电极与商用 Ag/AgCl 凝胶电极进行心电信号质量对比。方案属于设备原型验证，**非临床试验**，系统为**科研原型机，不用于临床诊断**。

### 伦理审批状态

**伦理审批在稿件撰写时尚未完成。** 研究方案、知情同意书及相关材料已依照《赫尔辛基宣言》及天津大学相关机构指南编写，将通过天津大学简化伦理审查路径（或导师备案的内部记录，具体路径待与导师团队确认）提交。审批通过后，将在本文及稿件中补充审批编号及审批日期。

**在获得伦理审批之前，本研究不会启动，不会收集任何人体数据。**

### 知情同意

所有潜在受试者将在实验开始前至少 5 分钟获得完整知情同意书（中文版，见 `docs/hardware/consent_form_zh.md`），内容涵盖研究目的与流程（共约 35 分钟，每种电极 6 个动作段）、潜在风险、数据匿名化方案（受试者编号 S01–S10，仅记录性别与年龄）、数据使用方式（学术论文聚合统计；原始匿名波形作为补充材料公开）、无条件退出权以及课题组与学术伦理委员会联系方式。仅签署知情同意书且所有疑问已获解答的受试者予以纳入。

---

**⚠️ Pre-submission action required:**
1. Obtain ethics approval (or waiver/exemption) from TJU.
2. Insert approval identifier and date above.
3. Confirm the approval path with Prof. Yang Hui (杨辉) and Prof. Huang Xian (黄显).
4. Update `docs/hardware/consent_form_zh.md` with advisor names and contact details before use.
5. If TJU determines that a formal IRB review is not required (e.g., low-risk device prototype validation with non-vulnerable adults), replace "ethics approval pending" with the appropriate waiver statement and reference.
