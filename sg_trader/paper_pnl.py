from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Position:
    quantity: float
    avg_price: float


def _positions_path(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "paper_positions.json"


def read_positions(cache_dir: Path) -> dict[str, Position]:
    path = _positions_path(cache_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    positions = {}
    for symbol, values in payload.items():
        try:
            quantity = float(values.get("quantity", 0.0))
            avg_price = float(values.get("avg_price", 0.0))
        except (TypeError, ValueError):
            continue
        positions[symbol] = Position(quantity=quantity, avg_price=avg_price)
    return positions


def write_positions(cache_dir: Path, positions: dict[str, Position]) -> None:
    path = _positions_path(cache_dir)
    payload = {
        symbol: {"quantity": pos.quantity, "avg_price": pos.avg_price}
        for symbol, pos in positions.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def apply_fill(
    cache_dir: Path, symbol: str, side: str, quantity: float, price: float
) -> Position:
    positions = read_positions(cache_dir)
    pos = positions.get(symbol, Position(quantity=0.0, avg_price=0.0))
    signed_qty = quantity if side.upper() == "BUY" else -quantity
    new_qty = pos.quantity + signed_qty
    realized_pnl = 0.0

    if pos.quantity == 0 or (pos.quantity > 0 and new_qty > 0) or (
        pos.quantity < 0 and new_qty < 0
    ):
        total_cost = pos.avg_price * pos.quantity + price * signed_qty
        if new_qty != 0:
            avg_price = total_cost / new_qty
        else:
            avg_price = 0.0
    else:
        closed_qty = min(abs(signed_qty), abs(pos.quantity))
        direction = 1 if pos.quantity > 0 else -1
        realized_pnl = (price - pos.avg_price) * closed_qty * direction
        avg_price = price if new_qty != 0 else 0.0

    updated = Position(quantity=new_qty, avg_price=avg_price)
    positions[symbol] = updated
    write_positions(cache_dir, positions)
    return updated, realized_pnl


def compute_unrealized_pnl(position: Position, mark_price: float) -> float:
    return (mark_price - position.avg_price) * position.quantity
