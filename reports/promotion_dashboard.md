# Promotion Dashboard

## Promotion Changelog
- v11_6_tuned -> v11_7_tuned: delta_terminal=0.422683, decision=PROMOTED_V11_7_TUNED

## Full-Sample Side-by-Side
    profile           mode  terminal_wealth   sharpe     maxdd
v11_6_tuned           base        28.609161 1.092790 -0.277268
v11_6_tuned with_overrides        32.525377 1.098449 -0.265694
v11_7_tuned           base        28.609161 1.092790 -0.277268
v11_7_tuned with_overrides        32.948060 1.101703 -0.267355

## OOT Selection (Top Row)
- source: v11_7b_local_wealth_refine_run1_scan.csv
 rank  stress_cap  derisk_cap  panic_cap  overall_scale  risk_on_scale  derisk_scale  panic_scale  min_median_terminal  min_pass_rate_terminal  avg_full_terminal  avg_full_sharpe  worst_fold_maxdd  worst_stress_maxdd  full_candidate_maxdd  fold_margin  stress_margin  full_margin  min_margin  wealth_dd_ratio
    1        0.16        0.38       0.06           1.01           0.98          1.01          0.8             1.664119                     1.0          27.425815         1.040051         -0.109633           -0.252148             -0.269219     0.110367       0.013852     0.001781    0.001781       101.871657

## Simulation Mode
- simulation_mode: tail_hedge_aware
- baseline_with_overrides_terminal_legacy_reference: 39.839243
- current_with_overrides_terminal_legacy_reference: 40.721934

## Active Smoke
               mode  terminal_wealth   sharpe     maxdd
      base_standard        28.609161 1.092790 -0.277268
with_overrides_auto        32.948060 1.101703 -0.267355

## Active State
- active_profile: v11_7_tuned
- active_label: run1_tight_weak_v224_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol120_vf000

## Decision
- PROMOTED_V11_7_TUNED
