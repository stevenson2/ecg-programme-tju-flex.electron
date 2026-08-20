# Graphical Abstract — Concept and Design Description

> **MDPI *Sensors* encourages a graphical abstract alongside the manuscript.**  
> This file describes the concept; the actual image should be generated as `graphical_abstract.png` (or `.tiff`) at ≥300 dpi.

---

## Concept

A single-panel schematic that communicates the **full system integration** (top half) and the **key patient-level evaluation result** (bottom half), visually establishing that this is a *system paper* with *rigorous evaluation* — not an algorithm-only or electrode-only contribution.

## Layout (Suggested: 1200 × 600 px at 300 dpi, horizontal)

### Top Half — System Schematic (left-to-right flow)

```
┌──────────┐   ┌──────────┐   ┌──────────────────────┐   ┌──────────┐   ┌──────────┐
│ Flexible │   │  AD8232  │   │   ESP32-S3 (Dual-Core)  │   │ BLE NUS  │   │ Flutter  │
│ PEDOT:   │──▶│   AFE    │──▶│  ┌─────────┬─────────┐│──▶│  Link    │──▶│   App    │
│ PSS Dry  │   │          │   │  │ Core 1  │ Core 0  ││   │          │   │          │
│ Electrode│   │          │   │  │Filter+HR│TFLite AI││   │          │   │ Real-time│
└──────────┘   └──────────┘   │  └─────────┴─────────┘│   └──────────┘   │ Waveform │
                              │  Dual-Expert:         │                  │ + AI Flag│
                              │  Arrhythmia OR MI     │                  └──────────┘
                              └──────────────────────┘
```

With small inset photos/icons: electrode film photo, ESP32-S3 board photo, phone screenshot.

### Bottom Half — Key Result (horizontal bar chart or compact table)

A compact visualization of the headline patient-level result:

| Model (exp6 patient-clean) | PTB-AUC | MIT-AUC |
|---------------------------|:-------:|:-------:|
| Training chain (D0)        | **0.8232** | 0.8245 |
| Deployment chain (D3, SGD) | 0.7697 | **0.9122** |

With a small callout: *"Patient-level 60/20/20 split, train∩test = ∅, AAMI per-class breakdown"*

And a small ΔAUC annotation: *"Deployment-chain mismatch cost: ΔAUC −0.105 → recovered to −0.054 via matched training"*

---

## Alternative: Dual-Panel Layout

If the horizontal layout is too cramped, use two stacked panels:

**Panel A (top 60%):** System block diagram with flow arrows, as above.

**Panel B (bottom 40%):** A 2×2 grid of compact results:
  1. PTB patient-level AUC bar (D0 vs. D3)
  2. MIT patient-level AUC bar
  3. AAMI per-class recall radar/spider chart (N, SVEB, VEB, F, Q) for the best model
  4. Competitor comparison: this work vs. [1]–[5] on a radar chart (axes: MCU-deployed, patient-level eval, dual-expert, flexible electrode, full-stack)

---

## Visual Style

- Color palette: medical-device blue (#1B4F72) and signal green (#27AE60), white/gray background
- Sans-serif font (Arial or Helvetica)
- No excessive text; labels in English
- Icons from a consistent icon set or hand-drawn schematic style
- 300 dpi minimum for MDPI submission

## Generation

The graphical abstract can be generated with one of:
1. **matplotlib + patches** (Python) — programmatic layout, reproducible
2. **draw.io / diagrams.net** — export to PNG, then upscale to 300 dpi
3. **Inkscape / Adobe Illustrator** — manual vector design for highest quality

For option 1, a generation script (`generate_graphical_abstract.py`) could be placed in `pc_tools/` using matplotlib's `FancyBboxPatch`, `FancyArrowPatch`, and `Rectangle` primitives.

---

**Note:** This is a concept description. The actual image file (`graphical_abstract.png`) should be generated and placed in this folder before submission. MDPI accepts graphical abstracts as PNG, JPEG, TIFF, or PDF; minimum resolution 300 dpi; recommended size 1200 × 600 pixels.
