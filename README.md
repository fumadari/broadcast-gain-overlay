# Broadcast–Gain (2‑byte) — CPU‑only Paper Pack

This repo runs a compact set of **CPU‑only** experiments to demonstrate a 2‑byte
**Broadcast–Gain** overlay for many‑agent coordination and produces **paper‑ready** figures.

**Highlights**
- Tiny overlay: 2 bytes/agent @ 15 Hz (≈0.24 kbps/agent payload).  
- Coordinate‑wise **median** fuse with TTL; robust to drops.  
- **Move/Wait‑only gating** of action logits (prevents excessive throttling).  
- **CDE** (congestion delay) computed via **free‑flow intent replay** in a ghost environment:  
  \( \mathrm{CDE} = (t_{\text{total}} - t_{\text{free}}) / t_{\text{total}} \).  
  This mirrors DEEPFLEET’s metric and motivation. (We use an intent‑based analog here.)

## Quick start
```bash
python main.py --mode quick --out results/
# just aggregate and make figures from existing CSVs:
python main.py --aggregate --out results/
```

## Figures produced
- `return_vs_dropout_(bernoulli|burst).(png|pdf)`  
- `cde_vs_dropout_(bernoulli|burst).(png|pdf)`  
- `tdvar_vs_dropout_bernoulli.(png|pdf)`  
- `alpha_beta_heatmap_improvement.(png|pdf)`  
- `radius_ablation.(png|pdf)`  
- `one_byte_tradeoff.(png|pdf)`  
- `gain_distribution.(png|pdf)`  
- `space_time_congestion.(png|pdf)`  
- `theory_varratio.(png|pdf)`, `median_vs_mean_robustness.(png|pdf)`

## Notes
- **Free‑flow / CDE**: our CDE uses intents recorded *before* arbitration and compares realized
  movement vs ghost free‑flow (no exclusivity). This is aligned with the metric definition used
  in the DEEPFLEET paper (§4, Table 2).

- Only `numpy`, `pandas`, and `matplotlib` are required.
