# Wealth-Constrained Retune Sweep

Generated: 2026-02-19 07:53:46

## Objective
- Search exposure combinations while enforcing wealth viability guardrails.

## Grid
- rerisk values: 0.2
- hold values: 0.0
- cap values: 0.05
- derisk values: -0.41,-0.05,0.0
- labels evaluated: 1

## Guardrails
- terminal_wealth > 0.4
- max_drawdown > -0.99

## Summary
- total combinations: 3
- viable combinations: 2

Top 2 viable combinations:

| rank_viable | label | rerisk | hold | cap | derisk | terminal_wealth | max_drawdown | avg_daily_ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | run1_tight_weak_v224_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol041_vf000 | 0.2 | 0.0 | 0.05 | 0.0 | 1.0 | 0.0 | 0.0 |
| 2 | run1_tight_weak_v224_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol041_vf000 | 0.2 | 0.0 | 0.05 | -0.05 | 0.41127860510502645 | -0.6120656046959837 | -6.256533437517435e-05 |
