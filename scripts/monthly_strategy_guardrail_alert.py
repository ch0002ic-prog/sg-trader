#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check monthly strategy guardrails and optionally send alert.",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Path to monthly_strategy_summary.json produced by monthly_strategy_summary.py",
    )
    parser.add_argument(
        "--soft-fail",
        action="store_true",
        help="Print breach details but always exit 0.",
    )
    return parser.parse_args()


def _send_telegram_alert(message: str) -> bool:
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False

    params = urllib.parse.urlencode({"chat_id": chat_id, "text": message})
    url = f"https://api.telegram.org/bot{token}/sendMessage?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_json)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    concentration = payload.get("concentration", [])
    breaches = [row for row in concentration if row.get("guardrail") == "BREACH"]

    if not breaches:
        print("[monthly-guardrail] OK: no guardrail breach")
        return 0

    lines = ["[monthly-guardrail] BREACH detected:"]
    for row in breaches:
        lines.append(
            "- "
            f"{row.get('profile')}: top3={float(row.get('top3_concentration', 0.0)):.4f}, "
            f"effective_n={float(row.get('effective_n', 0.0)):.2f}"
        )
    message = "\n".join(lines)
    print(message)

    sent = _send_telegram_alert(message)
    if sent:
        print("[monthly-guardrail] Telegram alert sent")
    else:
        print("[monthly-guardrail] Telegram alert not sent (missing creds or request failed)")

    if args.soft_fail:
        return 0
    return 8


if __name__ == "__main__":
    raise SystemExit(main())
