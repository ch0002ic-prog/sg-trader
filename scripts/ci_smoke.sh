#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

LEDGER_PATH="${LEDGER_PATH:-$ROOT_DIR/fortress_alpha_ledger.json}"
EXECUTION_BROKER="${EXECUTION_BROKER:-paper}"
PAPER_SYMBOL="${PAPER_SYMBOL:-SPX_PUT}"
PAPER_SIDE="${PAPER_SIDE:-SELL}"
PAPER_QTY="${PAPER_QTY:-1}"
PAPER_REFERENCE_PRICE="${PAPER_REFERENCE_PRICE:-1.25}"
PAPER_SEED="${PAPER_SEED:-1}"

ROBUSTNESS_STRATEGY_PROFILE="${ROBUSTNESS_STRATEGY_PROFILE:-aggressive}"
ROBUSTNESS_MAX_TOP3_CONCENTRATION="${ROBUSTNESS_MAX_TOP3_CONCENTRATION:-0.70}"
ROBUSTNESS_MIN_EFFECTIVE_N="${ROBUSTNESS_MIN_EFFECTIVE_N:-4.0}"
ROBUSTNESS_THRESHOLD_MODE="${ROBUSTNESS_THRESHOLD_MODE:-global}"
ROBUSTNESS_MAX_TOP3_CONCENTRATION_NORMAL="${ROBUSTNESS_MAX_TOP3_CONCENTRATION_NORMAL:-$ROBUSTNESS_MAX_TOP3_CONCENTRATION}"
ROBUSTNESS_MIN_EFFECTIVE_N_NORMAL="${ROBUSTNESS_MIN_EFFECTIVE_N_NORMAL:-$ROBUSTNESS_MIN_EFFECTIVE_N}"
ROBUSTNESS_MAX_TOP3_CONCENTRATION_DEFENSIVE="${ROBUSTNESS_MAX_TOP3_CONCENTRATION_DEFENSIVE:-$ROBUSTNESS_MAX_TOP3_CONCENTRATION}"
ROBUSTNESS_MIN_EFFECTIVE_N_DEFENSIVE="${ROBUSTNESS_MIN_EFFECTIVE_N_DEFENSIVE:-$ROBUSTNESS_MIN_EFFECTIVE_N}"

ENFORCE_ROBUSTNESS_GATE="${ENFORCE_ROBUSTNESS_GATE:-1}"
CI_SMOKE_SUMMARY_PATH="${CI_SMOKE_SUMMARY_PATH:-}"

if [[ ! -f "$ROOT_DIR/main.py" ]]; then
  echo "main.py not found in project root: $ROOT_DIR" >&2
  exit 2
fi

HEALTH_OUT=""
HEALTH_RC=0
ALLOCATOR_OUT=""
ALLOCATOR_RC=0
SMOKE_OUT=""
SMOKE_RC=0

ROBUSTNESS_DETECTED_REGIME=""
ROBUSTNESS_EFFECTIVE_MAX_TOP3_CONCENTRATION="$ROBUSTNESS_MAX_TOP3_CONCENTRATION"
ROBUSTNESS_EFFECTIVE_MIN_EFFECTIVE_N="$ROBUSTNESS_MIN_EFFECTIVE_N"

HEALTH_TMP="$(mktemp)"
ALLOCATOR_TMP="$(mktemp)"
ALLOCATOR_REPORT_TMP="$(mktemp)"
SMOKE_TMP="$(mktemp)"

cleanup() {
  rm -f "$HEALTH_TMP" "$ALLOCATOR_TMP" "$ALLOCATOR_REPORT_TMP" "$SMOKE_TMP"
}
trap cleanup EXIT

HEALTH_OUT="$($PYTHON main.py --ledger-path "$LEDGER_PATH" --healthcheck-json 2>&1)"
HEALTH_RC=$?
printf '%s' "$HEALTH_OUT" > "$HEALTH_TMP"

if [[ $HEALTH_RC -eq 0 ]]; then
  if [[ "$ENFORCE_ROBUSTNESS_GATE" == "1" ]]; then
    if [[ "$ROBUSTNESS_THRESHOLD_MODE" == "regime" ]]; then
      PREFLIGHT_REPORT_TMP="$(mktemp)"
      $PYTHON main.py \
        --ledger-path "$LEDGER_PATH" \
        --strategy-profile "$ROBUSTNESS_STRATEGY_PROFILE" \
        --regime-aware-defaults \
        --no-log \
        --report-path "$PREFLIGHT_REPORT_TMP" \
        >/dev/null 2>&1
      PREFLIGHT_RC=$?

      if [[ $PREFLIGHT_RC -eq 0 ]]; then
        ROBUSTNESS_DETECTED_REGIME="$($PYTHON - "$PREFLIGHT_REPORT_TMP" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

regime = str(payload.get("regime", {}).get("regime", "")).strip().lower()
if regime in {"normal", "defensive"}:
    print(regime)
else:
    print("")
PY
)"
      fi

      rm -f "$PREFLIGHT_REPORT_TMP"

      if [[ "$ROBUSTNESS_DETECTED_REGIME" == "defensive" ]]; then
        ROBUSTNESS_EFFECTIVE_MAX_TOP3_CONCENTRATION="$ROBUSTNESS_MAX_TOP3_CONCENTRATION_DEFENSIVE"
        ROBUSTNESS_EFFECTIVE_MIN_EFFECTIVE_N="$ROBUSTNESS_MIN_EFFECTIVE_N_DEFENSIVE"
      else
        ROBUSTNESS_EFFECTIVE_MAX_TOP3_CONCENTRATION="$ROBUSTNESS_MAX_TOP3_CONCENTRATION_NORMAL"
        ROBUSTNESS_EFFECTIVE_MIN_EFFECTIVE_N="$ROBUSTNESS_MIN_EFFECTIVE_N_NORMAL"
      fi
    fi

    ALLOCATOR_OUT="$($PYTHON main.py \
      --ledger-path "$LEDGER_PATH" \
      --strategy-profile "$ROBUSTNESS_STRATEGY_PROFILE" \
      --regime-aware-defaults \
      --no-log \
      --robustness-gate \
      --robustness-max-top3-concentration "$ROBUSTNESS_EFFECTIVE_MAX_TOP3_CONCENTRATION" \
      --robustness-min-effective-n "$ROBUSTNESS_EFFECTIVE_MIN_EFFECTIVE_N" \
      --report-path "$ALLOCATOR_REPORT_TMP" 2>&1)"
    ALLOCATOR_RC=$?
  fi

  if [[ $ALLOCATOR_RC -eq 0 ]]; then
    SMOKE_OUT="$($PYTHON main.py \
      --execution-ci-smoke \
      --execution-ci-smoke-json \
      --execution-broker "$EXECUTION_BROKER" \
      --paper-symbol "$PAPER_SYMBOL" \
      --paper-side "$PAPER_SIDE" \
      --paper-qty "$PAPER_QTY" \
      --paper-reference-price "$PAPER_REFERENCE_PRICE" \
      --paper-seed "$PAPER_SEED" 2>&1)"
    SMOKE_RC=$?
  fi
fi

printf '%s' "$ALLOCATOR_OUT" > "$ALLOCATOR_TMP"
printf '%s' "$SMOKE_OUT" > "$SMOKE_TMP"

SUMMARY="$($PYTHON - \
  "$HEALTH_RC" \
  "$ALLOCATOR_RC" \
  "$SMOKE_RC" \
  "$HEALTH_TMP" \
  "$ALLOCATOR_TMP" \
  "$SMOKE_TMP" \
  "$ENFORCE_ROBUSTNESS_GATE" \
  "$ALLOCATOR_REPORT_TMP" \
  "$ROBUSTNESS_THRESHOLD_MODE" \
  "$ROBUSTNESS_DETECTED_REGIME" \
  "$ROBUSTNESS_EFFECTIVE_MAX_TOP3_CONCENTRATION" \
  "$ROBUSTNESS_EFFECTIVE_MIN_EFFECTIVE_N" <<'PY'
import json
import sys
from pathlib import Path

health_rc = int(sys.argv[1])
allocator_rc = int(sys.argv[2])
smoke_rc = int(sys.argv[3])
health_raw = Path(sys.argv[4]).read_text(encoding="utf-8")
allocator_raw = Path(sys.argv[5]).read_text(encoding="utf-8")
smoke_raw = Path(sys.argv[6]).read_text(encoding="utf-8")
enforce_robustness_gate = sys.argv[7] == "1"
allocator_report_path = Path(sys.argv[8])
robustness_threshold_mode = sys.argv[9]
robustness_detected_regime = sys.argv[10]
robustness_effective_max_top3 = float(sys.argv[11])
robustness_effective_min_effective_n = float(sys.argv[12])


def parse_json(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return None


health_payload = parse_json(health_raw)
allocator_payload = None
if allocator_report_path.exists() and allocator_report_path.stat().st_size > 0:
    try:
        allocator_payload = json.loads(allocator_report_path.read_text(encoding="utf-8"))
    except Exception:
        allocator_payload = parse_json(allocator_raw)
else:
    allocator_payload = parse_json(allocator_raw)

smoke_payload = parse_json(smoke_raw)

summary = {
    "ok": health_rc == 0 and allocator_rc == 0 and smoke_rc == 0,
    "robustness_gate_enforced": enforce_robustness_gate,
    "robustness_threshold_mode": robustness_threshold_mode,
    "robustness_detected_regime": robustness_detected_regime or None,
    "robustness_effective_thresholds": {
        "max_top3_concentration": robustness_effective_max_top3,
        "min_effective_n": robustness_effective_min_effective_n,
    },
    "healthcheck": {
        "rc": health_rc,
        "payload": health_payload,
        "raw": None if health_payload is not None else health_raw,
    },
    "allocator_robustness": {
        "rc": allocator_rc,
        "payload": allocator_payload,
        "raw": None if allocator_payload is not None else allocator_raw,
    },
    "execution_ci_smoke": {
        "rc": smoke_rc,
        "payload": smoke_payload,
        "raw": None if smoke_payload is not None else smoke_raw,
    },
}

print(json.dumps(summary, sort_keys=True))
PY
)"

printf '%s\n' "$SUMMARY"

if [[ -n "$CI_SMOKE_SUMMARY_PATH" ]]; then
  mkdir -p "$(dirname "$CI_SMOKE_SUMMARY_PATH")"
  printf '%s\n' "$SUMMARY" > "$CI_SMOKE_SUMMARY_PATH"
fi

if [[ $HEALTH_RC -ne 0 ]]; then
  exit $HEALTH_RC
fi
if [[ $ALLOCATOR_RC -ne 0 ]]; then
  exit $ALLOCATOR_RC
fi
if [[ $SMOKE_RC -ne 0 ]]; then
  exit $SMOKE_RC
fi

exit 0
