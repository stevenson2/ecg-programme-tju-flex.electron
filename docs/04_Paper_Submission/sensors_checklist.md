# MDPI *Sensors* Submission Checklist

> **Date:** 2026-08-03  
> **Manuscript:** "A Flexible Dry-Electrode ECG System with Dual-Expert On-Device Deep Learning on an ESP32-S3 Microcontroller: System Design, Deployment-Chain-Matched Training, and Patient-Level Evaluation"  
> **Sources:** [Sensors Instructions for Authors](https://www.mdpi.com/journal/sensors/instructions) · [MDPI Author Layout Style Guide](https://www.mdpi.com/authors/layout) · [MDPI Research Data Policies](https://www.mdpi.com/ethics)  
> **Caveat:** Direct fetch of mdpi.com was blocked (403 bot-protection). Content was retrieved via search-engine index of the official pages. Cross-check against the live page at submission time.

---

## MDPI Official Submission Checklist (4 items)

| # | Requirement | Status | Notes |
|---|-------------|:------:|-------|
| 1 | Read the **Aims & Scope** and confirm suitability | ✅ | *Sensors* scope: wearable sensors, biomedical signal processing, AI-enabled sensing — all core to this paper. |
| 2 | Use the **Microsoft Word template** or **LaTeX template** | ⬜ | Manuscript currently in plain Markdown (`docs/manuscript_sections_1_4.md`). Must be converted to MDPI Word or LaTeX template before submission (blocked by todos 15/18). |
| 3 | Ensure **publication ethics, research ethics, copyright, authorship, figure formats, data and reference formats** are considered | ⬜ | Ethics pending TJU approval. Author contributions drafted. Figures pending (todo 12). References in MDPI numbered style needed at final assembly (todo 15). |
| 4 | All authors have **approved** the manuscript and **read the Instructions** | ⬜ | Co-authors (杨辉, 黄显) have not yet reviewed the manuscript draft. Must confirm before submission. |

---

## Detailed Item-by-Item Checklist

### A. Manuscript Format

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **Template** | Word or LaTeX (MDPI `mdpi` class) | ⬜ | Convert Markdown → MDPI template at final assembly (todo 15/18). Template was revised Dec 2025 for 2026 volumes — download current version. |
| **Section order** | Author Info → Abstract → Keywords → Introduction → Materials & Methods → Results → Discussion → Conclusions; back matter: Supplementary Materials, Author Contributions, Funding, Data Availability Statement, Acknowledgments, Conflicts of Interest, References | ⬜ | Current draft has Intro (1), Related Work (2), System Design (3), Methods (4); Results/Discussion/Conclusions in todo 14. Reorder at final assembly. |
| **Page/line numbers** | Continuous line numbers; page numbers per template | ⬜ | Template handles automatically. |

### B. Abstract

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **Length** | ~200 words maximum, single paragraph | ⬜ | Abstract not yet drafted — to be written at final assembly (todo 15) from the Introduction summary. |
| **Structure** | Background → Methods → Results → Conclusion (no headings) | ⬜ | Draft abstract following this structured flow. |
| **Restrictions** | No citations, no undefined abbreviations, no math formulas | ⬜ | Verify at final pass. |

### C. Keywords

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **Count** | 3 to 10 | ⬜ | Propose: flexible dry electrode, ECG, ESP32-S3, TFLite Micro, patient-level evaluation, deployment-chain matching, dual-expert fusion, arrhythmia detection, myocardial infarction, wearable sensor. |
| **Specificity** | Specific to the article, reasonably common in the discipline | ⬜ | Verify against *Sensors* keyword frequency. |

### D. Highlights

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **Inclusion** | **Optional** in *Sensors*; up to 2 bullet points per part (main findings, implications) | ⬜ | Drafted 5 highlights in `highlights.md`. MDPI general practice elsewhere is 3–5 bullets ≤85 characters each. Trim or reformat per Sensors-specific instruction if needed. |
| **Content** | Not a copy of the abstract; substantive findings | ⬜ | Highlights drafted are substantive and distinct from what an abstract would say. |

### E. References

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **Style** | Numbered in order of appearance, square brackets [1], [1–3], [1,3] | ⬜ | LITERATURE_MATRIX.md entries use matrix number order; must be renumbered by order of appearance at final assembly (todo 15). |
| **Placement** | Before punctuation | ⬜ | Verify throughout. |
| **Journal names** | **Full journal names** (no abbreviations) — standard MDPI style for *Sensors* | ⬜ | Matrix uses abbreviated names in some entries (e.g., "IEEE TBME" → "IEEE Transactions on Biomedical Engineering"). Expand at final assembly. |
| **Author format** | Surname, Initials.; semicolons between authors | ⬜ | Verify at final assembly. |
| **DOIs** | Required where available; highly encouraged | ✅ | All 59 matrix entries with verified DOIs listed. |
| **Bibliography software** | EndNote, Zotero, Mendeley recommended | ⬜ | Set up MDPI output style in chosen tool. |
| **Supplementary citations** | Allowed in supplementary files if also in main reference list | ⬜ | Cross-check supplementary S-files' citations against main reference list. |

### F. Figures

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **Resolution** | ≥600 dpi for line art; ~300 dpi for photos | ⬜ | Figures not yet generated (todo 12). Ensure ≥600 dpi for block diagrams and schematics. |
| **Format** | PNG, JPEG, TIFF | ⬜ | Export figures in these formats; avoid PDF figures. |
| **Placement** | Insert near first citation; numbered sequentially | ⬜ | F1–F6 in Sections 3; F7+ in Sections 5-7 (todo 14). Plan insertion points. |
| **Panels** | Labeled a/b/c; scale bars marked | ⬜ | Ensure all multi-panel figures have sub-labels. |
| **Fonts** | Embedded; English text only; correct math symbols | ⬜ | Use embedded fonts; auditable by opening figure in a viewer. |
| **Editable parts** | No editable parts in figure files | ⬜ | Flatten layers; rasterize text. |
| **Graphical abstract** | Min. 560 × 1100 px; PNG/JPEG/TIFF; original (not reused from figures) | ⬜ | Concept described in `graphical_abstract.md`. Generate actual image. |

### G. Supplementary Materials

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **Citation** | Must be cited in main text with S-prefix: Figure S1, Table S1, Equation S2... | ⬜ | Add S-citations to manuscript at final assembly; cross-reference `supplementary_notes.md`. |
| **Format** | Any format; common non-proprietary recommended | ⬜ | Use PDF/JSON/CSV/ZIP as appropriate. |
| **Hosting** | Upload with manuscript OR third-party repository with DOI + preservation policy | ⬜ | Decide: upload as .zip with submission, or host on Zenodo with DataCite DOI. |

### H. Data Availability Statement

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **Requirement** | **Required for ALL MDPI articles** | ✅ | Drafted in `data_availability.md` using the "contained within article/supplementary + on request" template. |
| **Template match** | Should match one of MDPI's 6 suggested statements | ⬜ | Current draft combines two templates. Verify the combined wording is accepted; if not, split or choose one template. |
| **No data created** | Even if no new data, a statement is still required | N/A | Not our case — data from public datasets + code available. |

### I. English Language

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **Quality** | Grammatically correct English required | ⬜ | Manuscript draft in review stage. Final English polish before submission (MDPI editing service available as backup — APC covers only minor editing). |
| **Polish** | APC includes minor English editing only | ⬜ | Self-review thoroughly; use MDPI editing only if needed. |

### J. Ethics and Declarations

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **IRB approval** | Required for studies involving humans | ⬜ | **Pending TJU ethics review.** See `ethics_statement.md`. Will not submit until approved or waiver obtained. |
| **Informed consent** | Required for studies involving humans | ⬜ | Consent form template drafted (`docs/hardware/consent_form_zh.md`). Will obtain signed consent before data collection. |
| **Conflicts of interest** | Must be declared | ⬜ | No conflicts to declare — to be confirmed with all authors. |
| **Author contributions** | CRediT taxonomy recommended | ⬜ | Drafted in `author_contributions.md`. Must be confirmed with co-authors. |
| **Funding** | Must be declared | ⬜ | Funding sources to be confirmed with Prof. Yang Hui. |

### K. Submission System Requirements

| Item | MDPI Requirement | Status | Action |
|------|------------------|:------:|--------|
| **Cover letter** | Required | ✅ | Drafted in `cover_letter.md`. Fill author emails, reviewer suggestions, and date before uploading. |
| **Suggested reviewers** | Optional | ⬜ | Placeholder in cover letter. Discuss with advisors. |
| **ORCID** | Recommended for all authors | ⬜ | Collect ORCID iDs from co-authors. |

---

## Pre-Submission Action Items (Prioritized)

1. **[BLOCKING]** Obtain ethics approval or waiver from Tianjin University. The human-subject protocol cannot proceed, and the paper cannot be submitted, until this is resolved.
2. **[BLOCKING]** Convert manuscript to MDPI Word or LaTeX template (todo 15/18).
3. **[BLOCKING]** Generate all figures at ≥600 dpi (todo 12).
4. **[BLOCKING]** Obtain co-author review and approval of the manuscript, author contributions, and cover letter.
5. Renumber references by order of appearance; expand journal names to full form.
6. Write abstract (~200 words, structured style).
7. Confirm keywords with co-authors.
8. Polish English throughout.
9. Generate graphical abstract image.
10. Prepare and upload supplementary materials.
11. Fill all bracketed placeholders in cover letter (author emails, affiliations, date).
12. Confirm data availability statement wording matches an accepted MDPI template.

---

**Sources verified:**
- [Sensors Instructions for Authors](https://www.mdpi.com/journal/sensors/instructions)
- [MDPI Author Layout Style Guide](https://www.mdpi.com/authors/layout)
- [MDPI Research Data Policies](https://www.mdpi.com/ethics)

*Generated 2026-08-03 by task 16. Cross-check against the live pages before submission.*
