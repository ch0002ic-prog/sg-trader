#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

if [[ ! -f "$ROOT_DIR/main.py" ]]; then
  echo "main.py not found in project root: $ROOT_DIR" >&2
  exit 2
fi

"$PYTHON" -m unittest \
  tests.test_healthcheck \
  tests.test_cli_capabilities \
  tests.test_local_ci_cli_args \
  tests.test_local_ci_parse \
  tests.test_print_ci_artifact_summary \
  tests.test_execution_replay_json \
  tests.test_execution_ci_smoke \
  tests.test_execution_id_only \
  tests.test_list_brokers \
  tests.test_execution_replay \
  tests.test_portfolio_dashboard_skip_recent_cli
