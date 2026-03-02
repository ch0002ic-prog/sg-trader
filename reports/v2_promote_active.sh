#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEL="$REPO_ROOT/reports/v2_promotion_selection.json"

if [[ ! -f "$SEL" ]]; then
  echo "missing selection file: $SEL" >&2
  exit 1
fi

winner_label=$(python3 -c 'import json, pathlib; p=pathlib.Path("reports/v2_promotion_selection.json"); print(json.loads(p.read_text())["winner"]["label"])')

echo "$winner_label" > "$REPO_ROOT/reports/active_profile_label.txt"
cp "$REPO_ROOT/reports/v2_active_policy_overrides.csv" "$REPO_ROOT/reports/active_policy_overrides.csv"

echo "updated reports/active_profile_label.txt"
echo "updated reports/active_policy_overrides.csv"
