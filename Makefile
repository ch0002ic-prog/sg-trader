test:
	python -m unittest discover -s tests

.PHONY: promotion-dashboard switch-v6-tuned rollback-v5 smoke-active daily-check daily-check-enforce-v8 daily-check-enforce-v10 daily-check-enforce-v11-2 daily-check-enforce-v11-3 daily-check-enforce-v11-4 daily-check-enforce-v11-5 daily-check-enforce-v11-6 daily-check-enforce-v11-7 v7-scan switch-v7-tuned rollback-v6-tuned v8-scan switch-v8-tuned rollback-v7-tuned v9-scan switch-v9-tuned rollback-v8-tuned v10-scan switch-v10-tuned rollback-v9-tuned v11-1-scan v11-2-scan v11-3-scan v11-4-scan v11-5-scan v11-6-scan v11-7-scan switch-v11-2-tuned rollback-v10-tuned rollback-v10-verify switch-v11-2-verify switch-v11-3-tuned switch-v11-3-verify rollback-v11-2-tuned rollback-v11-2-verify switch-v11-4-tuned switch-v11-4-verify rollback-v11-3-tuned rollback-v11-3-verify switch-v11-5-tuned switch-v11-5-verify switch-v11-6-tuned switch-v11-6-verify switch-v11-7-tuned switch-v11-7-verify rollback-v11-3-from-v11-5 rollback-v11-3-from-v11-5-verify rollback-v11-5-from-v11-6 rollback-v11-5-from-v11-6-verify rollback-v11-6-from-v11-7 rollback-v11-6-from-v11-7-verify

PYTHON ?= /Users/ch0002techvc/Downloads/sg-trader/.venv/bin/python

v9-repro:
	$(PYTHON) main2.py --tickers-from-config all --min-drop-sharpe 0.3 --min-drop-cagr 0.03 --include-vix-regime --policy-min-events 50 --policy-min-events-equity 50 --policy-min-events-vol 40 --oos-split-date 2019-01-01 --run-label run1_tight_weak_v9_regimeboost --policy-rerisk-vol 1.0 --policy-hold-vol 0.8 --policy-cap-vol 0.5 --policy-derisk-vol 0.0 --cooldown-days 5 --cooldown-dd-extra 5 --equity-hold-boost 1.05 --archive-run
	$(PYTHON) main2.py --tickers-from-config all --min-drop-sharpe 0.3 --min-drop-cagr 0.03 --include-vix-regime --policy-min-events 50 --policy-min-events-equity 50 --policy-min-events-vol 40 --oos-split-date 2019-01-01 --run-label run1_tight_weak_v9_ddhard --policy-rerisk-vol 1.0 --policy-hold-vol 0.8 --policy-cap-vol 0.5 --policy-derisk-vol 0.0 --cooldown-days 5 --cooldown-dd-extra 5 --dd-hard-stop --archive-run
	$(PYTHON) main2.py --tickers-from-config all --min-drop-sharpe 0.3 --min-drop-cagr 0.03 --include-vix-regime --policy-min-events 50 --policy-min-events-equity 50 --policy-min-events-vol 40 --oos-split-date 2019-01-01 --run-label run1_tight_weak_v9_volfloor --policy-rerisk-vol 1.0 --policy-hold-vol 0.8 --policy-cap-vol 0.5 --policy-derisk-vol 0.0 --cooldown-days 5 --cooldown-dd-extra 5 --vol-floor 0.12 --vol-floor-boost 1.05 --archive-run

v9-summary:
	$(PYTHON) -c "from pathlib import Path; import pandas as pd; root=Path('reports/selected_table_checks'); labels={'v9_regimeboost':'overlay_summary_run1_tight_weak_v9_regimeboost.csv','v9_ddhard':'overlay_summary_run1_tight_weak_v9_ddhard.csv','v9_volfloor':'overlay_summary_run1_tight_weak_v9_volfloor.csv'}; [print(k, f'sharpe={full[\"sharpe\"]:.4f}', f'cagr={full[\"cagr\"]:.2%}', f'vol={full[\"vol\"]:.2%}', f'max_dd={full[\"max_dd\"]:.2%}') for k,f in labels.items() for full in [(pd.read_csv(root/f).query('segment == \"full\"')[['sharpe','cagr','vol','max_dd']].mean())]]"

v9-next-sweep:
	$(PYTHON) main2.py --tickers-from-config all --min-drop-sharpe 0.3 --min-drop-cagr 0.03 --include-vix-regime --policy-min-events 50 --policy-min-events-equity 50 --policy-min-events-vol 40 --oos-split-date 2019-01-01 --run-label run1_tight_weak_v9_tune_w45 --policy-rerisk-vol 1.0 --policy-hold-vol 0.8 --policy-cap-vol 0.5 --policy-derisk-vol 0.0 --cooldown-days 5 --cooldown-dd-extra 5 --window-days 45 --vol-floor 0.12 --vol-floor-boost 1.05
	$(PYTHON) main2.py --tickers-from-config all --min-drop-sharpe 0.3 --min-drop-cagr 0.03 --include-vix-regime --policy-min-events 50 --policy-min-events-equity 50 --policy-min-events-vol 40 --oos-split-date 2019-01-01 --run-label run1_tight_weak_v9_tune_w60 --policy-rerisk-vol 1.0 --policy-hold-vol 0.8 --policy-cap-vol 0.5 --policy-derisk-vol 0.0 --cooldown-days 5 --cooldown-dd-extra 5 --window-days 60 --vol-floor 0.12 --vol-floor-boost 1.05
	$(PYTHON) main2.py --tickers-from-config all --min-drop-sharpe 0.3 --min-drop-cagr 0.03 --include-vix-regime --policy-min-events 50 --policy-min-events-equity 50 --policy-min-events-vol 40 --oos-split-date 2019-01-01 --run-label run1_tight_weak_v9_tune_trend005 --policy-rerisk-vol 1.0 --policy-hold-vol 0.8 --policy-cap-vol 0.5 --policy-derisk-vol 0.0 --cooldown-days 5 --cooldown-dd-extra 5 --trend-strength-pct 0.005 --vol-floor 0.12 --vol-floor-boost 1.05

promotion-dashboard:
	$(PYTHON) scripts/v6_promotion_dashboard.py

switch-v6-tuned:
	bash reports/v6_switch_to_tuned.sh

rollback-v5:
	bash reports/v6_switch_to_v5.sh

smoke-active:
	$(PYTHON) scripts/smoke_active_profile.py

daily-check:
	$(PYTHON) scripts/run_daily_check.py

daily-check-enforce-v8:
	$(PYTHON) scripts/run_daily_check.py --enforce-v8

daily-check-enforce-v10:
	$(PYTHON) scripts/run_daily_check.py --enforce-v10

daily-check-enforce-v11-2:
	$(PYTHON) scripts/run_daily_check.py --enforce-v11-2

daily-check-enforce-v11-3:
	$(PYTHON) scripts/run_daily_check.py --enforce-v11-3

daily-check-enforce-v11-4:
	$(PYTHON) scripts/run_daily_check.py --enforce-v11-4

daily-check-enforce-v11-5:
	$(PYTHON) scripts/run_daily_check.py --enforce-v11-5

daily-check-enforce-v11-6:
	$(PYTHON) scripts/run_daily_check.py --enforce-v11-6

daily-check-enforce-v11-7:
	$(PYTHON) scripts/run_daily_check.py --enforce-v11-7

v7-scan:
	$(PYTHON) scripts/v7_promotion_workbench.py --out-prefix reports/v7_promotion_workbench_run1

switch-v7-tuned:
	bash reports/v7_switch_to_tuned.sh

rollback-v6-tuned:
	bash reports/v7_switch_to_v6_tuned.sh

v8-scan:
	$(PYTHON) scripts/v8_promotion_workbench.py --folds 4 --cost-grid 2,4 --out-prefix reports/v8_promotion_workbench_run1

v9-scan:
	$(PYTHON) scripts/v9_promotion_workbench.py --folds 4 --cost-grid 2,4 --persistence-shifts -1,0,1 --out-prefix reports/v9_promotion_workbench_run1

v10-scan:
	$(PYTHON) scripts/v10_promotion_workbench.py --folds 4 --cost-grid 2,4 --persistence-shifts -1,0,1 --maxdd-cap -0.22 --out-prefix reports/v10_promotion_workbench_run1

v11-1-scan:
	$(PYTHON) scripts/v11_1_risk_refit_workbench.py --folds 4 --cost-grid 2,4,6 --persistence-days 1 --out-prefix reports/v11_1_risk_refit_run1

v11-2-scan:
	$(PYTHON) scripts/v11_2_tail_hedge_workbench.py --folds 4 --cost-grid 2,4,6 --persistence-days 1 --out-prefix reports/v11_2_tail_hedge_run1

v11-3-scan:
	$(PYTHON) scripts/v11_3_tail_hedge_workbench.py --folds 4 --cost-grid 2,4,6,8,10 --persistence-days 1 --out-prefix reports/v11_3_tail_hedge_run1

v11-4-scan:
	$(PYTHON) scripts/v11_4_robust_frontier_workbench.py --folds 4 --cost-grid 2,4,6,8,10 --persistence-days 1 --out-prefix reports/v11_4_robust_frontier_run1

v11-5-scan:
	$(PYTHON) scripts/v11_5_wealth_unlock_workbench.py --folds 4 --cost-grid 2,4,6,8,10 --persistence-days 1 --out-prefix reports/v11_5_wealth_unlock_run1

v11-6-scan:
	$(PYTHON) scripts/v11_6_local_wealth_refine_workbench.py --folds 4 --cost-grid 2,4,6,8,10 --persistence-days 1 --out-prefix reports/v11_6_local_wealth_refine_run1

v11-7-scan:
	$(PYTHON) scripts/v11_7_local_wealth_refine_workbench.py --folds 4 --cost-grid 2,4,6,8,10 --persistence-days 1 --out-prefix reports/v11_7_local_wealth_refine_run1

v11-scan:
	$(PYTHON) scripts/v11_promotion_workbench.py --folds 4 --cost-grid 2,4,6 --persistence-shifts -1,0,1 --maxdd-cap -0.22 --full-maxdd-cap -0.26 --out-prefix reports/v11_promotion_workbench_run1

switch-v8-tuned:
	bash reports/v8_switch_to_tuned.sh

rollback-v7-tuned:
	bash reports/v8_switch_to_v7_tuned.sh

switch-v9-tuned:
	bash reports/v9_switch_to_tuned.sh

rollback-v8-tuned:
	bash reports/v9_switch_to_v8_tuned.sh

switch-v10-tuned:
	bash reports/v10_switch_to_tuned.sh

rollback-v9-tuned:
	bash reports/v10_switch_to_v9_tuned.sh

switch-v11-2-tuned:
	bash reports/v11_2_switch_to_tuned.sh

switch-v11-2-verify:
	bash reports/v11_2_switch_to_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

switch-v11-3-tuned:
	bash reports/v11_3_switch_to_tuned.sh

switch-v11-3-verify:
	bash reports/v11_3_switch_to_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

switch-v11-4-tuned:
	bash reports/v11_4_switch_to_tuned.sh

switch-v11-4-verify:
	bash reports/v11_4_switch_to_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

switch-v11-5-tuned:
	bash reports/v11_5_switch_to_tuned.sh

switch-v11-5-verify:
	bash reports/v11_5_switch_to_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

switch-v11-6-tuned:
	bash reports/v11_6_switch_to_tuned.sh

switch-v11-6-verify:
	bash reports/v11_6_switch_to_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

switch-v11-7-tuned:
	bash reports/v11_7_switch_to_tuned.sh

switch-v11-7-verify:
	bash reports/v11_7_switch_to_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

rollback-v10-tuned:
	bash reports/v11_2_switch_to_v10_tuned.sh

rollback-v11-2-tuned:
	bash reports/v11_3_switch_to_v11_2_tuned.sh

rollback-v11-2-verify:
	bash reports/v11_3_switch_to_v11_2_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

rollback-v11-3-tuned:
	bash reports/v11_4_switch_to_v11_3_tuned.sh

rollback-v11-3-verify:
	bash reports/v11_4_switch_to_v11_3_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

rollback-v11-3-from-v11-5:
	bash reports/v11_5_switch_to_v11_3_tuned.sh

rollback-v11-3-from-v11-5-verify:
	bash reports/v11_5_switch_to_v11_3_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

rollback-v11-5-from-v11-6:
	bash reports/v11_6_switch_to_v11_5_tuned.sh

rollback-v11-5-from-v11-6-verify:
	bash reports/v11_6_switch_to_v11_5_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

rollback-v11-6-from-v11-7:
	bash reports/v11_7_switch_to_v11_6_tuned.sh

rollback-v11-6-from-v11-7-verify:
	bash reports/v11_7_switch_to_v11_6_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py

rollback-v10-verify:
	bash reports/v11_2_switch_to_v10_tuned.sh
	$(PYTHON) scripts/smoke_active_profile.py
