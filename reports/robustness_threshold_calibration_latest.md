# Robustness Threshold Calibration (Latest)

- Runs: 36
- Profiles: aggressive, defensive, normal
- Lookbacks: 21, 63

## Distribution Summary

- top3 concentration: min=0.6271, p50=0.7143, p80=0.8644, p90=1.0000, max=1.0000
- effective_n: min=2.0000, p10=2.0000, p20=3.4433, p50=4.3654, max=5.4139

## Recommended Defaults

- ROBUSTNESS_MAX_TOP3_CONCENTRATION=0.8644
- ROBUSTNESS_MIN_EFFECTIVE_N=3.4433
- Method: set to p80(top3) and p20(effective_n) from current calibration grid

## Worst Top-3 Concentration Scenarios

- normal_default_penalties_minnone_lb21: top3=1.0000, effective_n=2.0000, selected=2
- normal_default_penalties_min0p0_lb21: top3=1.0000, effective_n=2.0000, selected=2
- normal_default_penalties_min0p05_lb21: top3=1.0000, effective_n=2.0000, selected=2
- defensive_default_penalties_minnone_lb21: top3=1.0000, effective_n=2.0000, selected=2
- defensive_default_penalties_min0p0_lb21: top3=1.0000, effective_n=2.0000, selected=2

## Worst Effective-N Scenarios

- normal_default_penalties_minnone_lb21: effective_n=2.0000, top3=1.0000, selected=2
- normal_default_penalties_min0p0_lb21: effective_n=2.0000, top3=1.0000, selected=2
- normal_default_penalties_min0p05_lb21: effective_n=2.0000, top3=1.0000, selected=2
- defensive_default_penalties_minnone_lb21: effective_n=2.0000, top3=1.0000, selected=2
- defensive_default_penalties_min0p0_lb21: effective_n=2.0000, top3=1.0000, selected=2
