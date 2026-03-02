from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AllocationSnapshot:
    fortress: float
    alpha: float
    shield: float


TARGET_ALLOCATIONS = AllocationSnapshot(fortress=0.70, alpha=0.29, shield=0.01)


def _normalize_allocations(snapshot: AllocationSnapshot) -> AllocationSnapshot:
    total = snapshot.fortress + snapshot.alpha + snapshot.shield
    if total <= 0:
        return snapshot
    if total > 1.5:
        return AllocationSnapshot(
            fortress=snapshot.fortress / 100.0,
            alpha=snapshot.alpha / 100.0,
            shield=snapshot.shield / 100.0,
        )
    if abs(total - 1.0) > 0.05:
        return AllocationSnapshot(
            fortress=snapshot.fortress / total,
            alpha=snapshot.alpha / total,
            shield=snapshot.shield / total,
        )
    return snapshot


def check_allocation_drift(
    snapshot: AllocationSnapshot, band: float
) -> tuple[str | None, dict[str, float]]:
    normalized = _normalize_allocations(snapshot)
    drifts = {
        "fortress": normalized.fortress - TARGET_ALLOCATIONS.fortress,
        "alpha": normalized.alpha - TARGET_ALLOCATIONS.alpha,
        "shield": normalized.shield - TARGET_ALLOCATIONS.shield,
    }
    breaches = {
        key: value for key, value in drifts.items() if abs(value) > band
    }
    if not breaches:
        return None, drifts

    lines = ["REBALANCE DRIFT ALERT"]
    for key, value in breaches.items():
        current = getattr(normalized, key)
        target = getattr(TARGET_ALLOCATIONS, key)
        lines.append(
            f"- {key.title()}: {current:.2%} (target {target:.2%}, drift {value:+.2%})"
        )
    return "\n".join(lines), drifts
