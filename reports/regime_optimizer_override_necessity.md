# Override Necessity Ablation

For each run: baseline vs full overrides, leave-one-out drops, and grouped drops.

## v224_safe
      run                                   scenario  terminal_wealth  delta_vs_full_wealth   sharpe  delta_vs_full_sharpe     maxdd  delta_vs_full_maxdd
v224_safe                      baseline_no_overrides         1.004404             -0.417488 0.057288             -0.569748 -0.006880         2.491695e-02
v224_safe                        drop_group::de-risk         1.041060             -0.380832 0.202905             -0.424131 -0.010702         2.109488e-02
v224_safe drop::low|trend_off|dd_ok|vol_low::de-risk         1.142034             -0.279857 0.475330             -0.151706 -0.031797        -9.992007e-16
v224_safe drop::mid|trend_off|dd_ok|vol_low::de-risk         1.296193             -0.125699 0.495141             -0.131895 -0.028000         3.797015e-03
v224_safe                           drop_group::hold         1.371904             -0.049988 0.602153             -0.024883 -0.033315        -1.517164e-03
v224_safe    drop::low|trend_off|dd_ok|vol_low::hold         1.384489             -0.037403 0.620725             -0.006311 -0.031797        -7.771561e-16
v224_safe    drop::mid|trend_off|dd_ok|vol_low::hold         1.408965             -0.012927 0.609686             -0.017350 -0.033315        -1.517164e-03
v224_safe                             full_overrides         1.421892              0.000000 0.627036              0.000000 -0.031797         0.000000e+00

## v232_safe
      run                                   scenario  terminal_wealth  delta_vs_full_wealth   sharpe  delta_vs_full_sharpe     maxdd  delta_vs_full_maxdd
v232_safe                      baseline_no_overrides         1.023765             -0.405381 0.181507             -0.413254 -0.012137         3.294081e-02
v232_safe                        drop_group::de-risk         1.066565             -0.362581 0.211612             -0.383148 -0.024157         2.092068e-02
v232_safe drop::low|trend_off|dd_ok|vol_low::de-risk         1.147865             -0.281281 0.392718             -0.202042 -0.045077         6.661338e-16
v232_safe drop::mid|trend_off|dd_ok|vol_low::de-risk         1.327941             -0.101204 0.494774             -0.099986 -0.030343         1.473466e-02
v232_safe                           drop_group::hold         1.371903             -0.057243 0.602131              0.007371 -0.033315         1.176288e-02
v232_safe    drop::low|trend_off|dd_ok|vol_low::hold         1.391555             -0.037591 0.584171             -0.010589 -0.045077         1.110223e-15
v232_safe    drop::mid|trend_off|dd_ok|vol_low::hold         1.408964             -0.020181 0.609664              0.014903 -0.033315         1.176288e-02
v232_safe                             full_overrides         1.429146              0.000000 0.594760              0.000000 -0.045077         0.000000e+00

