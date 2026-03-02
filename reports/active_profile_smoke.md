# Active Profile Smoke

- active_label: run1_tight_weak_v224_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol120_vf000
- profile: v11_7_tuned
- exposures: rerisk=0.04, hold=0.46, cap=0.03, derisk=0.18, persistence_days=1
- override_states: 9

## Results
               mode  terminal_wealth   sharpe     maxdd
      base_standard        28.609161 1.092790 -0.277268
with_overrides_auto        32.948060 1.101703 -0.267355

## Decision
- smoke_status: PASS
