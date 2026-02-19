#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path):
    if not path.exists():
        return None, f"missing:{path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid:{path} ({exc})"


def _as_bool(value):
    return bool(value)


def _yn(value):
    return "yes" if value else "no"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print one-line status summary from CI artifact JSON files."
    )
    parser.add_argument(
        "--smoke-path",
        default="reports/ci_smoke_summary.json",
        help="Path to ci_smoke summary JSON (default: reports/ci_smoke_summary.json)",
    )
    parser.add_argument(
        "--local-ci-path",
        default="reports/local_ci_result.json",
        help="Path to local CI result JSON (default: reports/local_ci_result.json)",
    )
    args = parser.parse_args()

    smoke_payload, smoke_error = _load_json(Path(args.smoke_path))
    local_payload, local_error = _load_json(Path(args.local_ci_path))

    smoke_ok = _as_bool(smoke_payload.get("ok")) if smoke_payload else False
    local_ok = _as_bool(local_payload.get("ok")) if local_payload else False

    smoke_rc = smoke_payload.get("healthcheck", {}).get("rc", "na") if smoke_payload else "na"
    smoke_exec_rc = smoke_payload.get("execution_ci_smoke", {}).get("rc", "na") if smoke_payload else "na"
    local_overall_rc = local_payload.get("overall_rc", "na") if local_payload else "na"
    local_strict = local_payload.get("strict", "na") if local_payload else "na"

    errors = [e for e in [smoke_error, local_error] if e]
    status_ok = smoke_ok and local_ok and not errors
    status = "PASS" if status_ok else "FAIL"

    line = (
        "ci-artifacts "
        f"status={status} "
        f"smoke_ok={_yn(smoke_ok)} "
        f"smoke_health_rc={smoke_rc} "
        f"smoke_exec_rc={smoke_exec_rc} "
        f"local_ok={_yn(local_ok)} "
        f"local_strict={local_strict} "
        f"local_overall_rc={local_overall_rc}"
    )

    if errors:
        line += " errors=" + "|".join(errors)

    print(line)
    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
