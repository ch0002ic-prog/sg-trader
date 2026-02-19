# Regime Fine Consensus Candidates

Source files: 4
Total consensus rows: 2
Filtered candidates: 2

## Top candidates

| regime_combo | rs_bin | focus_action | against_action | version_count | mean_edge | min_edge | min_focus_n | min_against_n |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mid|trend_off|dd_ok|vol_low | rs_q3 | hold | de-risk | 4 | 0.012743 | 0.012743 | 70 | 420 |
| mid|trend_off|dd_ok|vol_low | rs_q4 | hold | de-risk | 4 | 0.001057 | 0.001057 | 88 | 345 |

## Experimental Validation (2026-02-18)

- Compared existing fine-override variants vs family baselines using OOS deltas and day-to-day action deltas.
- run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_finespxq3safe: sharpe_diff=-0.000001, cagr_diff=2.072616, action_change_rate=0.000000.
- run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000: sharpe_diff=-0.000002, cagr_diff=1.438625, action_change_rate=0.000000.
- run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_finespx: sharpe_diff=-0.040255, cagr_diff=-88.292478, action_change_rate=0.020725.
- run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineq3: sharpe_diff=-0.057673, cagr_diff=-615.458704, action_change_rate=0.017709.
- run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr: sharpe_diff=-0.087582, cagr_diff=-944.341150, action_change_rate=0.032909.
- Result: no tested fine-override variant shows robust portfolio-level improvement; keep fine overrides disabled by default.
