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
CI_SMOKE_SUMMARY_PATH="${CI_SMOKE_SUMMARY_PATH:-}"

if [[ ! -f "$ROOT_DIR/main.py" ]]; then
  echo "main.py not found in project root: $ROOT_DIR" >&2
  exit 2
fi

HEALTH_OUT=""
HEALTH_RC=0
SMOKE_OUT=""
SMOKE_RC=0
HEALTH_TMP="$(mktemp)"
SMOKE_TMP="$(mktemp)"

cleanup() {
  rm -f "$HEALTH_TMP" "$SMOKE_TMP"
}
trap cleanup EXIT

HEALTH_OUT="$($PYTHON main.py --ledger-path "$LEDGER_PATH" --healthcheck-json 2>&1)"
HEALTH_RC=$?
printf '%s' "$HEALTH_OUT" > "$HEALTH_TMP"

if [[ $HEALTH_RC -eq 0 ]]; then
  SMOKE_OUT="$($PYTHON main.py --execution-ci-smoke --execution-ci-smoke-json \
    --execution-broker "$EXECUTION_BROKER" \
    --paper-symbol "$PAPER_SYMBOL" \
    --paper-side "$PAPER_SIDE" \
    --paper-qty "$PAPER_QTY" \
    --paper-reference-price "$PAPER_REFERENCE_PRICE" \
    --paper-seed "$PAPER_SEED" 2>&1)"
  SMOKE_RC=$?
fi
printf '%s' "$SMOKE_OUT" > "$SMOKE_TMP"

SUMMARY="$($PYTHON - <<'PY' "$HEALTH_RC" "$SMOKE_RC" "$HEALTH_TMP" "$SMOKE_TMP"
import json
import sys
from pathlib import Path

health_rc = int(sys.argv[1])
smoke_rc = int(sys.argv[2])
health_raw = Path(sys.argv[3]).read_text(encoding="utf-8")
smoke_raw = Path(sys.argv[4]).read_text(encoding="utf-8")

def parse_json(raw):
    try:
        return json.loads(raw)
    except Exception:
        return None

health_payload = parse_json(health_raw)
smoke_payload = parse_json(smoke_raw)
summary = {
    "ok": health_rc == 0 and smoke_rc == 0,
    "healthcheck": {
        "rc": health_rc,
        "payload": health_payload,
        "raw": None if health_payload is not None else health_raw,
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
if [[ $SMOKE_RC -ne 0 ]]; then
  exit $SMOKE_RC
fi
exit 0
