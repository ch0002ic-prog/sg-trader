cd /Users/ch0002techvc/Downloads/sg-trader && base_cmd(){
  /Users/ch0002techvc/Downloads/sg-trader/.venv/bin/python main2.py \
    --tickers-from-config all \
    --min-drop-sharpe 0.3 \
    --min-drop-cagr 0.03 \
    --include-vix-regime \
    --policy-min-events 50 \
    --policy-min-events-equity 50 \
    --policy-min-events-vol 40 \
    --oos-split-date 2019-01-01 \
    --policy-rerisk-vol 1.0 \
    --policy-hold-vol 0.8 \
    --policy-cap-vol 0.5 \
    --policy-derisk-vol 0.0 \
    --policy-rerisk-vol-trend-off 0.69 \
    --policy-hold-vol-trend-off 0.47 \
    --policy-cap-vol-trend-off 0.15 \
    --policy-derisk-vol-trend-off 0.0 \
    --policy-rerisk-vix-trend-off 0.686 \
    --policy-hold-vix-trend-off 0.466 \
    --policy-cap-vix-trend-off 0.146 \
    --policy-derisk-vix-trend-off 0.0 \
    --policy-rerisk-vvix-trend-off 0.683 \
    --policy-hold-vvix-trend-off 0.463 \
    --policy-cap-vvix-trend-off 0.143 \
    --policy-derisk-vvix-trend-off 0.0 \
    --cooldown-days 22 \
    --cooldown-dd-extra 3 \
    --vol-floor 0.11 \
    --vol-floor-boost 1.05 \
    "$@"
}

# 1) Event gates
for pme in 40 50 60; do
  for pmeq in 40 50 60; do
    for pmev in 30 40 50; do
      if [[ "$pme" == "50" && "$pmeq" == "50" && "$pmev" == "40" ]]; then continue; fi
      # keep compact: only 4 curated combos
      key="${pme}_${pmeq}_${pmev}"
      [[ "$key" == "40_40_30" || "$key" == "40_50_30" || "$key" == "60_60_50" || "$key" == "60_50_50" ]] || continue
      label="run1_tight_weak_v41_evt_pme${pme}_eq${pmeq}_vol${pmev}"
      echo "==> $label"
      base_cmd --run-label "$label" --policy-min-events "$pme" --policy-min-events-equity "$pmeq" --policy-min-events-vol "$pmev" || exit 1
    done
  done
done

# 2) Quantiles (vix x vol)
for vixq in 0.70 0.75 0.80; do
  for volq in 0.70 0.75 0.80; do
    if [[ "$vixq" == "0.75" && "$volq" == "0.75" ]]; then continue; fi
    label="run1_tight_weak_v41_q_vix${vixq/./}_vol${volq/./}"
    echo "==> $label"
    base_cmd --run-label "$label" --vix-quantile "$vixq" --vol-quantile "$volq" || exit 1
  done
done

# 3) Vol quantile window
for vqw in 126 252 504; do
  label="run1_tight_weak_v41_vqw_${vqw}"
  echo "==> $label"
  base_cmd --run-label "$label" --vol-quantile-window "$vqw" || exit 1
done

# 4) Drawdown threshold x hard stop
for ddt in 0.08 0.10 0.12; do
  for h in true false; do
    [[ "$ddt" == "0.10" && "$h" == "false" ]] && continue
    label="run1_tight_weak_v41_dd_ddt${ddt/./}_hs${h}"
    echo "==> $label"
    if [[ "$h" == "true" ]]; then
      base_cmd --run-label "$label" --drawdown-threshold "$ddt" --dd-hard-stop || exit 1
    else
      base_cmd --run-label "$label" --drawdown-threshold "$ddt" --no-dd-hard-stop || exit 1
    fi
  done
done