#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
py="$repo_root/.venv/bin/python"

cp "$repo_root/reports/v7_tuned_overrides.csv" "$repo_root/reports/active_policy_overrides.csv"
"$py" - <<'PY'
import json
from pathlib import Path

reports = Path('reports')
profile = json.loads((reports / 'v7_tuned_profile.json').read_text(encoding='utf-8'))
(reports / 'active_profile_exposures.json').write_text(json.dumps(profile, indent=2) + '\n', encoding='utf-8')
(reports / 'active_profile_label.txt').write_text(str(profile['label']) + '\n', encoding='utf-8')
print('activated', profile.get('profile', 'v7_tuned'))
PY
