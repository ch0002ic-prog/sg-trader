#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STRICT_MODE=0
JSON_MODE=0

usage() {
	cat <<'EOF'
Usage: bash scripts/local_ci.sh [--strict] [--json]

Options:
	--strict   Validate smoke JSON summary structure and require ok=true before unit gates.
	--json     Emit one final JSON status line for tooling.
	-h, --help Show this help message.
EOF
}

log() {
	if [[ $JSON_MODE -eq 0 ]]; then
		echo "$1"
	fi
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--strict)
			STRICT_MODE=1
			shift
			;;
		--json)
			JSON_MODE=1
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

if [[ -n "${PYTHON_BIN:-}" ]]; then
	PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
	PYTHON="$ROOT_DIR/.venv/bin/python"
else
	PYTHON="python3"
fi

SMOKE_SUMMARY_PATH=""
if [[ $STRICT_MODE -eq 1 ]]; then
	SMOKE_SUMMARY_PATH="$(mktemp)"
	trap 'rm -f "$SMOKE_SUMMARY_PATH"' EXIT
fi

SMOKE_RC=0
STRICT_VALIDATION_RC=0
UNIT_GATES_RC=0

log "[1/2] Running smoke gate..."
if [[ $STRICT_MODE -eq 1 ]]; then
	if [[ $JSON_MODE -eq 1 ]]; then
		CI_SMOKE_SUMMARY_PATH="$SMOKE_SUMMARY_PATH" bash scripts/ci_smoke.sh >/dev/null 2>&1 || SMOKE_RC=$?
	else
		CI_SMOKE_SUMMARY_PATH="$SMOKE_SUMMARY_PATH" bash scripts/ci_smoke.sh || SMOKE_RC=$?
	fi
	if [[ $SMOKE_RC -eq 0 ]]; then
		if [[ $JSON_MODE -eq 1 ]]; then
			"$PYTHON" - <<'PY' "$SMOKE_SUMMARY_PATH" >/dev/null 2>&1 || STRICT_VALIDATION_RC=$?
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
	raise SystemExit("Strict mode failed: smoke summary file not created")

try:
	payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
	raise SystemExit(f"Strict mode failed: invalid smoke JSON ({exc})")

if "ok" not in payload:
	raise SystemExit("Strict mode failed: smoke JSON missing 'ok' field")
if payload.get("ok") is not True:
	raise SystemExit("Strict mode failed: smoke JSON ok != true")
PY
		else
			"$PYTHON" - <<'PY' "$SMOKE_SUMMARY_PATH" || STRICT_VALIDATION_RC=$?
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
	raise SystemExit("Strict mode failed: smoke summary file not created")

try:
	payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
	raise SystemExit(f"Strict mode failed: invalid smoke JSON ({exc})")

if "ok" not in payload:
	raise SystemExit("Strict mode failed: smoke JSON missing 'ok' field")
if payload.get("ok") is not True:
	raise SystemExit("Strict mode failed: smoke JSON ok != true")
print("Strict mode: smoke summary validated (ok=true)")
PY
		fi
	fi
else
	if [[ $JSON_MODE -eq 1 ]]; then
		bash scripts/ci_smoke.sh >/dev/null 2>&1 || SMOKE_RC=$?
	else
		bash scripts/ci_smoke.sh || SMOKE_RC=$?
	fi
fi

if [[ $SMOKE_RC -eq 0 && $STRICT_VALIDATION_RC -eq 0 ]]; then
	log "[2/2] Running unit gates..."
	if [[ $JSON_MODE -eq 1 ]]; then
		bash scripts/unit_gates.sh >/dev/null 2>&1 || UNIT_GATES_RC=$?
	else
		bash scripts/unit_gates.sh || UNIT_GATES_RC=$?
	fi
fi

OVERALL_RC=0
if [[ $SMOKE_RC -ne 0 ]]; then
	OVERALL_RC=$SMOKE_RC
elif [[ $STRICT_VALIDATION_RC -ne 0 ]]; then
	OVERALL_RC=$STRICT_VALIDATION_RC
elif [[ $UNIT_GATES_RC -ne 0 ]]; then
	OVERALL_RC=$UNIT_GATES_RC
fi

if [[ $JSON_MODE -eq 1 ]]; then
	"$PYTHON" - <<'PY' "$STRICT_MODE" "$SMOKE_RC" "$STRICT_VALIDATION_RC" "$UNIT_GATES_RC" "$OVERALL_RC"
import json
import sys

strict = int(sys.argv[1]) == 1
smoke_rc = int(sys.argv[2])
strict_validation_rc = int(sys.argv[3])
unit_gates_rc = int(sys.argv[4])
overall_rc = int(sys.argv[5])

payload = {
	"ok": overall_rc == 0,
	"strict": strict,
	"smoke_rc": smoke_rc,
	"strict_validation_rc": strict_validation_rc,
	"unit_gates_rc": unit_gates_rc,
	"overall_rc": overall_rc,
}
print(json.dumps(payload, sort_keys=True))
PY
	exit $OVERALL_RC
fi

if [[ $OVERALL_RC -ne 0 ]]; then
	echo "Local CI failed (smoke_rc=$SMOKE_RC, strict_validation_rc=$STRICT_VALIDATION_RC, unit_gates_rc=$UNIT_GATES_RC)" >&2
elif [[ $STRICT_MODE -eq 1 ]]; then
	echo "Local CI passed (strict): smoke + unit-gates"
else
	echo "Local CI passed: smoke + unit-gates"
fi

exit $OVERALL_RC
