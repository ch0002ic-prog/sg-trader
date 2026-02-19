#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${MONTHLY_STRATEGY_OUT_DIR:-/tmp/sg_trader_monthly}"
WINDOWS="${MONTHLY_STRATEGY_WINDOWS:-10}"
FORWARD_DAYS="${MONTHLY_STRATEGY_FORWARD_DAYS:-21}"
LOOKBACKS="${MONTHLY_STRATEGY_LOOKBACKS:-63,126,252}"
PROFILES="${MONTHLY_STRATEGY_PROFILES:-normal,defensive,aggressive}"
CURRENT_DEFAULT="${MONTHLY_STRATEGY_CURRENT_DEFAULT:-aggressive}"
SWITCH_CANDIDATE="${MONTHLY_STRATEGY_SWITCH_CANDIDATE:-defensive}"
TOP3_THRESHOLD="${MONTHLY_STRATEGY_TOP3_THRESHOLD:-0.70}"
EFFECTIVE_N_THRESHOLD="${MONTHLY_STRATEGY_EFFECTIVE_N_THRESHOLD:-4.50}"
SOFT_FAIL_GUARDRAIL="${MONTHLY_STRATEGY_SOFT_FAIL_GUARDRAIL:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
	PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
	PYTHON="$ROOT_DIR/.venv/bin/python"
else
	PYTHON="python3"
fi

usage() {
	cat <<'EOF'
Usage: bash scripts/monthly_strategy_check.sh [options]

Runs non-mutating monthly strategy evaluation and emits a decision pack.

Options:
	--out-dir PATH               Output directory (default: /tmp/sg_trader_monthly)
	--windows N                  Walk-forward windows (default: 10)
	--forward-days N             Forward horizon (default: 21)
	--lookbacks csv              Comma-separated lookbacks (default: 63,126,252)
	--profiles csv               Comma-separated profiles (default: normal,defensive,aggressive)
	--current-default PROFILE    Current default profile (default: aggressive)
	--switch-candidate PROFILE   Candidate profile for switch rule (default: defensive)
	--top3-threshold X           Guardrail threshold (default: 0.70)
	--effective-n-threshold X    Guardrail threshold (default: 4.50)
	--soft-fail-guardrail        Do not fail command on guardrail breach (prints alert only)
	-h, --help                   Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--out-dir)
			OUT_DIR="$2"
			shift 2
			;;
		--windows)
			WINDOWS="$2"
			shift 2
			;;
		--forward-days)
			FORWARD_DAYS="$2"
			shift 2
			;;
		--lookbacks)
			LOOKBACKS="$2"
			shift 2
			;;
		--profiles)
			PROFILES="$2"
			shift 2
			;;
		--current-default)
			CURRENT_DEFAULT="$2"
			shift 2
			;;
		--switch-candidate)
			SWITCH_CANDIDATE="$2"
			shift 2
			;;
		--top3-threshold)
			TOP3_THRESHOLD="$2"
			shift 2
			;;
		--effective-n-threshold)
			EFFECTIVE_N_THRESHOLD="$2"
			shift 2
			;;
		--soft-fail-guardrail)
			SOFT_FAIL_GUARDRAIL=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			usage >&2
			exit 2
			;;
	esac
done

mkdir -p "$OUT_DIR"

IFS=',' read -r -a LOOKBACK_ARRAY <<< "$LOOKBACKS"
IFS=',' read -r -a PROFILE_ARRAY <<< "$PROFILES"

echo "[monthly] output dir: $OUT_DIR"
echo "[monthly] lookbacks: $LOOKBACKS"
echo "[monthly] profiles: $PROFILES"

SCAN_ARGS=()
for raw_lb in "${LOOKBACK_ARRAY[@]}"; do
	lb="$(echo "$raw_lb" | xargs)"
	if [[ -z "$lb" ]]; then
		continue
	fi
	echo "[monthly] walk-forward scan lookback=$lb"
	"$PYTHON" scripts/walkforward_profile_scan.py \
		--profiles "$PROFILES" \
		--lookback-days "$lb" \
		--forward-days "$FORWARD_DAYS" \
		--windows "$WINDOWS" \
		--out-csv "$OUT_DIR/walkforward_profile_scan_lb${lb}.csv" \
		--out-md "$OUT_DIR/walkforward_profile_scan_lb${lb}.md" \
		--out-detail-csv "$OUT_DIR/walkforward_profile_scan_lb${lb}_detail.csv" >/dev/null
	SCAN_ARGS+=("--scan-csv" "${lb}=$OUT_DIR/walkforward_profile_scan_lb${lb}.csv")
done

ALLOC_ARGS=()
for raw_profile in "${PROFILE_ARRAY[@]}"; do
	profile="$(echo "$raw_profile" | xargs)"
	if [[ -z "$profile" ]]; then
		continue
	fi
	echo "[monthly] allocation snapshot profile=$profile"
	"$PYTHON" main.py \
		--strategy-profile "$profile" \
		--lookback-days 63 \
		--no-log \
		--report-path "$OUT_DIR/sg_trader_strategy_${profile}.json" >/dev/null
	ALLOC_ARGS+=("--allocation" "${profile}=$OUT_DIR/sg_trader_strategy_${profile}.json")
done

echo "[monthly] building decision summary"
"$PYTHON" scripts/monthly_strategy_summary.py \
	"${SCAN_ARGS[@]}" \
	"${ALLOC_ARGS[@]}" \
	--current-default "$CURRENT_DEFAULT" \
	--switch-candidate "$SWITCH_CANDIDATE" \
	--top3-threshold "$TOP3_THRESHOLD" \
	--effective-n-threshold "$EFFECTIVE_N_THRESHOLD" \
	--out-json "$OUT_DIR/monthly_strategy_summary.json" \
	--out-md "$OUT_DIR/monthly_strategy_summary.md"

GUARDRAIL_ARGS=()
if [[ "$SOFT_FAIL_GUARDRAIL" == "1" ]]; then
	GUARDRAIL_ARGS+=("--soft-fail")
fi

echo "[monthly] checking guardrail alerts"
"$PYTHON" scripts/monthly_strategy_guardrail_alert.py \
	--summary-json "$OUT_DIR/monthly_strategy_summary.json" \
	"${GUARDRAIL_ARGS[@]}"

echo "[monthly] summary files:"
echo "- $OUT_DIR/monthly_strategy_summary.md"
echo "- $OUT_DIR/monthly_strategy_summary.json"
