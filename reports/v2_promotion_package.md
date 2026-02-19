# V2 Promotion Package

## Winner
- label: run1_tight_weak_v224_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol120_vf000
- exposures: rerisk=0.2, hold=0.05, cap=0.05, derisk=0.0
- fold pass rate: 100%
- median validation terminal wealth: 1.131157
- median validation sharpe: 0.843777
- median validation max drawdown: -0.021653
- overrides source: reports/regime_optimizer_v224_ndvol120_safe_overrides.csv

## Fallback
- label: run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr
- exposures: rerisk=0.2, hold=0.05, cap=0.05, derisk=0.0
- fold pass rate: 100%
- median validation terminal wealth: 1.112827
- median validation sharpe: 0.756403
- median validation max drawdown: -0.024972
- overrides source: reports/regime_optimizer_v232_fineeqnr_safe_overrides.csv

## Artifacts
- selection json: reports/v2_promotion_selection.json
- active overrides csv: reports/v2_active_policy_overrides.csv
- leaderboard: reports/regime_optimizer_v2_safe_leaderboard.csv
- folds: reports/regime_optimizer_v2_safe_folds.csv

## Notes
- Winner is chosen by v2 walk-forward stability ranking.
- Override necessity ablation confirms overrides should remain enabled.
