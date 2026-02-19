#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def _load_payload(path: str | None):
    raw = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    raw = raw.strip()
    if not raw:
        raise ValueError("empty input")
    return json.loads(raw)


def _bool(value):
    return "yes" if bool(value) else "no"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse scripts/local_ci.sh --json output and print compact status."
    )
    parser.add_argument(
        "--input",
        help="Path to JSON payload file. If omitted, reads from stdin.",
    )
    args = parser.parse_args()

    try:
        payload = _load_payload(args.input)
    except Exception as exc:
        print(f"local-ci parse error: {exc}", file=sys.stderr)
        return 2

    required = [
        "ok",
        "strict",
        "smoke_rc",
        "strict_validation_rc",
        "unit_gates_rc",
        "overall_rc",
    ]
    missing = [name for name in required if name not in payload]
    if missing:
        print(f"local-ci parse error: missing keys: {', '.join(missing)}", file=sys.stderr)
        return 2

    overall_rc = int(payload.get("overall_rc", 1))
    status = "PASS" if bool(payload.get("ok")) and overall_rc == 0 else "FAIL"
    print(
        "local-ci "
        f"status={status} "
        f"strict={_bool(payload['strict'])} "
        f"smoke_rc={payload['smoke_rc']} "
        f"strict_validation_rc={payload['strict_validation_rc']} "
        f"unit_gates_rc={payload['unit_gates_rc']} "
        f"overall_rc={overall_rc}"
    )
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
