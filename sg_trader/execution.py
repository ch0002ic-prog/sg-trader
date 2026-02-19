from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Protocol


@dataclass
class ExecutionRequest:
    symbol: str
    side: str
    quantity: float
    reference_price: float
    slippage_bps: float = 5.0
    latency_ms: int = 150
    seed: int | None = None


@dataclass
class ExecutionResult:
    symbol: str
    side: str
    quantity: float
    reference_price: float
    fill_price: float
    slippage_bps: float
    latency_ms: int


class ExecutionBroker(Protocol):
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        ...


class ExecutionAdapter(Protocol):
    name: str

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        ...


class ExecutionRouter:
    def __init__(self) -> None:
        self._brokers: dict[str, ExecutionBroker] = {}

    def register(self, name: str, broker: ExecutionBroker) -> None:
        self._brokers[name] = broker

    def register_adapter(self, adapter: ExecutionAdapter) -> None:
        self._brokers[adapter.name] = adapter

    def list_brokers(self) -> list[str]:
        return sorted(self._brokers.keys())

    def execute(self, name: str, request: ExecutionRequest) -> ExecutionResult:
        broker = self._brokers.get(name)
        if broker is None:
            raise ValueError(f"Unknown broker: {name}")
        return broker.execute(request)


class PaperBroker:
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        rng = random.Random(request.seed)
        slip_factor = (request.slippage_bps / 10000.0) * rng.uniform(0.5, 1.5)
        if request.side.upper() == "BUY":
            fill_price = request.reference_price * (1 + slip_factor)
        else:
            fill_price = request.reference_price * (1 - slip_factor)
        if request.latency_ms > 0:
            time.sleep(request.latency_ms / 1000.0)
        return ExecutionResult(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            reference_price=request.reference_price,
            fill_price=fill_price,
            slippage_bps=request.slippage_bps,
            latency_ms=request.latency_ms,
        )


class ManualBroker:
    name = "manual"

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        raise RuntimeError(
            "Manual broker selected; execute outside the system and log the fill."
        )


class DryRunBroker:
    name = "dry-run"

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            reference_price=request.reference_price,
            fill_price=request.reference_price,
            slippage_bps=0.0,
            latency_ms=0,
        )


class ExternalBrokerStub:
    name = "external"

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        raise RuntimeError(
            "External broker adapter not configured; implement a live adapter."
        )
