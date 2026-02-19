# Wealth-Constrained Retune Sweep

Generated: 2026-02-19 08:31:08

## Objective
- Search exposure combinations while enforcing wealth viability guardrails.

## Grid
- rerisk values: 0.2,0.3,0.4,0.5
- hold values: 0.0,0.02,0.05
- cap values: 0.05,0.1,0.15
- derisk values: -0.4,-0.2,-0.1,-0.05,0.0
- labels evaluated: 538

## Guardrails
- terminal_wealth > 1.0
- max_drawdown > -0.99

## Summary
- total combinations: 96840
- viable combinations: 12912

Top 30 viable combinations:

| rank_viable | label | rerisk | hold | cap | derisk | terminal_wealth | max_drawdown | avg_daily_ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.2 | 0.05 | 0.05 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 2 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.2 | 0.05 | 0.1 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 3 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.2 | 0.05 | 0.15 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 4 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.3 | 0.05 | 0.05 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 5 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.3 | 0.05 | 0.1 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 6 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.3 | 0.05 | 0.15 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 7 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.4 | 0.05 | 0.05 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 8 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.4 | 0.05 | 0.1 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 9 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.4 | 0.05 | 0.15 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 10 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.5 | 0.05 | 0.05 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 11 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.5 | 0.05 | 0.1 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 12 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineeqnr | 0.5 | 0.05 | 0.15 | 0.0 | 1.023764643809064 | -0.012136618652304532 | 1.6782652643709747e-06 |
| 13 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.2 | 0.05 | 0.05 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 14 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.2 | 0.05 | 0.1 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 15 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.2 | 0.05 | 0.15 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 16 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.3 | 0.05 | 0.05 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 17 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.3 | 0.05 | 0.1 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 18 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.3 | 0.05 | 0.15 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 19 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.4 | 0.05 | 0.05 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 20 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.4 | 0.05 | 0.1 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 21 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.4 | 0.05 | 0.15 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 22 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.5 | 0.05 | 0.05 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 23 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.5 | 0.05 | 0.1 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 24 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol1181_vf000_fineovr2 | 0.5 | 0.05 | 0.15 | 0.0 | 1.0237645286047294 | -0.01213664914515955 | 1.6783764242311328e-06 |
| 25 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineq3 | 0.2 | 0.05 | 0.05 | 0.0 | 1.0234178368142894 | -0.009184893605803768 | 1.6515699375047888e-06 |
| 26 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineq3 | 0.2 | 0.05 | 0.1 | 0.0 | 1.0234178368142894 | -0.009184893605803768 | 1.6515699375047888e-06 |
| 27 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineq3 | 0.2 | 0.05 | 0.15 | 0.0 | 1.0234178368142894 | -0.009184893605803768 | 1.6515699375047888e-06 |
| 28 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineq3 | 0.3 | 0.05 | 0.05 | 0.0 | 1.0234178368142894 | -0.009184893605803768 | 1.6515699375047888e-06 |
| 29 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineq3 | 0.3 | 0.05 | 0.1 | 0.0 | 1.0234178368142894 | -0.009184893605803768 | 1.6515699375047888e-06 |
| 30 | run1_tight_weak_v232_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol11817_vf000_fineq3 | 0.3 | 0.05 | 0.15 | 0.0 | 1.0234178368142894 | -0.009184893605803768 | 1.6515699375047888e-06 |
