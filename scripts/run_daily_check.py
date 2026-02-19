from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run daily monitoring checks with optional profile enforcement")
    parser.add_argument("--enforce-v8", action="store_true", help="Switch active profile to v8_tuned before running checks")
    parser.add_argument("--enforce-v10", action="store_true", help="Switch active profile to v10_tuned before running checks")
    parser.add_argument("--enforce-v11-2", action="store_true", help="Switch active profile to v11_2_tuned before running checks")
    parser.add_argument("--enforce-v11-3", action="store_true", help="Switch active profile to v11_3_tuned before running checks")
    parser.add_argument("--enforce-v11-4", action="store_true", help="Switch active profile to v11_4_tuned before running checks")
    parser.add_argument("--enforce-v11-5", action="store_true", help="Switch active profile to v11_5_tuned before running checks")
    args = parser.parse_args()

    selected = int(args.enforce_v8) + int(args.enforce_v10) + int(args.enforce_v11_2) + int(args.enforce_v11_3) + int(args.enforce_v11_4) + int(args.enforce_v11_5)
    if selected > 1:
        raise ValueError("Use only one of --enforce-v8, --enforce-v10, --enforce-v11-2, --enforce-v11-3, --enforce-v11-4, or --enforce-v11-5")

    if args.enforce_v8:
        switch_script = REPORTS / "v8_switch_to_tuned.sh"
        if not switch_script.exists():
            raise FileNotFoundError(f"Missing switch script: {switch_script}")
        run(["bash", str(switch_script)])

    if args.enforce_v10:
        switch_script = REPORTS / "v10_switch_to_tuned.sh"
        if not switch_script.exists():
            raise FileNotFoundError(f"Missing switch script: {switch_script}")
        run(["bash", str(switch_script)])

    if args.enforce_v11_2:
        switch_script = REPORTS / "v11_2_switch_to_tuned.sh"
        if not switch_script.exists():
            raise FileNotFoundError(f"Missing switch script: {switch_script}")
        run(["bash", str(switch_script)])

    if args.enforce_v11_3:
        switch_script = REPORTS / "v11_3_switch_to_tuned.sh"
        if not switch_script.exists():
            raise FileNotFoundError(f"Missing switch script: {switch_script}")
        run(["bash", str(switch_script)])

    if args.enforce_v11_4:
        switch_script = REPORTS / "v11_4_switch_to_tuned.sh"
        if not switch_script.exists():
            raise FileNotFoundError(f"Missing switch script: {switch_script}")
        run(["bash", str(switch_script)])

    if args.enforce_v11_5:
        switch_script = REPORTS / "v11_5_switch_to_tuned.sh"
        if not switch_script.exists():
            raise FileNotFoundError(f"Missing switch script: {switch_script}")
        run(["bash", str(switch_script)])

    run([sys.executable, str(ROOT / "scripts" / "v6_promotion_dashboard.py")])
    run([sys.executable, str(ROOT / "scripts" / "smoke_active_profile.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
