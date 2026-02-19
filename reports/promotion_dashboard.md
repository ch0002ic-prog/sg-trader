# Promotion Dashboard

## Promotion Changelog
- v11_3_tuned -> v11_5_tuned: delta_terminal=4.008177, decision=PROMOTED_V11_5_TUNED

## Full-Sample Side-by-Side
    profile           mode  terminal_wealth   sharpe     maxdd
v11_3_tuned           base        22.715028 0.985009 -0.276224
v11_3_tuned with_overrides        32.834023 1.058787 -0.268332
v11_5_tuned           base        22.715028 0.985009 -0.276224
v11_5_tuned with_overrides        36.842200 1.068917 -0.268332

## OOT Selection (Top Row)
- source: v11_5_wealth_unlock_run1_scan.csv
 rank  stress_cap  derisk_cap  panic_cap  overall_scale  derisk_scale  panic_scale  min_median_terminal  min_pass_rate_terminal  avg_full_terminal  avg_full_sharpe  worst_fold_maxdd  worst_stress_maxdd  full_candidate_maxdd  fold_margin  stress_margin  full_margin  min_margin  wealth_dd_ratio
 1081        0.16        0.34       0.08            1.0           1.0         0.85             1.879293                     1.0          34.371036         1.048621         -0.109146           -0.253594             -0.268522     0.110854       0.012406     0.002478    0.002478       128.001048

## Simulation Mode
- simulation_mode: tail_hedge_aware
- baseline_with_overrides_terminal_legacy_reference: 57.576785
- current_with_overrides_terminal_legacy_reference: 57.576785

## Active Smoke
               mode  terminal_wealth   sharpe     maxdd
      base_standard        22.715028 0.985009 -0.276224
with_overrides_auto        36.842200 1.068917 -0.268332

## Active State
- active_profile: v11_5_tuned
- active_label: run1_tight_weak_v224_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol120_vf000

## Decision
- PROMOTED_V11_5_TUNED
