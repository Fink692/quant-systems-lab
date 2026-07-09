from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HawkesOrderFlowParams:
    base_intensity: np.ndarray
    excitation: np.ndarray
    decay: float
    volume: float = 1.0

    def validate(self) -> None:
        base = np.asarray(self.base_intensity, dtype=float)
        excitation = np.asarray(self.excitation, dtype=float)
        if base.shape != (2,) or excitation.shape != (2, 2):
            raise ValueError("base_intensity must have shape (2,) and excitation shape (2, 2)")
        if np.any(base < 0) or np.any(excitation < 0):
            raise ValueError("base_intensity and excitation must be non-negative")
        if self.decay <= 0 or self.volume <= 0:
            raise ValueError("decay and volume must be positive")
        if hawkes_stability_radius(excitation, self.decay) >= 1.0:
            raise ValueError("Hawkes branching spectral radius must be below one")

    @property
    def branching_matrix(self) -> np.ndarray:
        return hawkes_branching_matrix(self.excitation, self.decay)

    @property
    def branching_ratio(self) -> float:
        return hawkes_stability_radius(self.excitation, self.decay)


@dataclass(frozen=True)
class HawkesOrderFlowResult:
    events: pd.DataFrame
    horizon: float
    branching_matrix: np.ndarray
    branching_ratio: float

    @property
    def event_count(self) -> int:
        return int(len(self.events))

    @property
    def buy_count(self) -> int:
        if self.events.empty:
            return 0
        return int((self.events["side"] == "buy").sum())

    @property
    def sell_count(self) -> int:
        if self.events.empty:
            return 0
        return int((self.events["side"] == "sell").sum())

    @property
    def net_order_flow(self) -> float:
        if self.events.empty:
            return 0.0
        return float(self.events["signed_volume"].sum())

    @property
    def order_flow_imbalance(self) -> float:
        total = self.buy_count + self.sell_count
        return 0.0 if total == 0 else float((self.buy_count - self.sell_count) / total)

    @property
    def realized_intensity(self) -> float:
        return float(self.event_count / self.horizon)


def hawkes_branching_matrix(excitation: np.ndarray, decay: float) -> np.ndarray:
    """Return expected children per event for an exponential-kernel Hawkes process."""
    values = np.asarray(excitation, dtype=float)
    if values.shape != (2, 2):
        raise ValueError("excitation must have shape (2, 2)")
    if np.any(values < 0) or decay <= 0:
        raise ValueError("excitation must be non-negative and decay positive")
    return values / decay


def hawkes_stability_radius(excitation: np.ndarray, decay: float) -> float:
    """Spectral radius of the Hawkes branching matrix."""
    branching = hawkes_branching_matrix(excitation, decay)
    return float(np.max(np.abs(np.linalg.eigvals(branching))))


def simulate_hawkes_order_flow(
    params: HawkesOrderFlowParams,
    horizon: float,
    seed: int | None = None,
    max_events: int = 100_000,
) -> HawkesOrderFlowResult:
    """Simulate buy/sell market-order flow with bivariate exponential Hawkes dynamics."""
    params.validate()
    if horizon <= 0 or max_events < 1:
        raise ValueError("horizon must be positive and max_events at least one")

    base = np.asarray(params.base_intensity, dtype=float)
    excitation = np.asarray(params.excitation, dtype=float)
    rng = np.random.default_rng(seed)
    time = 0.0
    excitation_state = np.zeros(2, dtype=float)
    rows: list[dict[str, float | str]] = []

    while time < horizon and len(rows) < max_events:
        upper_intensity = float((base + excitation_state).sum())
        if upper_intensity <= 0.0:
            break
        dt = float(rng.exponential(1.0 / upper_intensity))
        time += dt
        excitation_state *= np.exp(-params.decay * dt)
        if time > horizon:
            break

        intensity = base + excitation_state
        total_intensity = float(intensity.sum())
        if rng.random() * upper_intensity > total_intensity:
            continue

        event_type = int(rng.choice(2, p=intensity / total_intensity))
        side = "buy" if event_type == 0 else "sell"
        signed_volume = params.volume if event_type == 0 else -params.volume
        rows.append(
            {
                "time": float(time),
                "side": side,
                "signed_volume": float(signed_volume),
                "buy_intensity": float(intensity[0]),
                "sell_intensity": float(intensity[1]),
            }
        )
        excitation_state += excitation[:, event_type]

    events = pd.DataFrame(rows, columns=["time", "side", "signed_volume", "buy_intensity", "sell_intensity"])
    return HawkesOrderFlowResult(
        events=events,
        horizon=float(horizon),
        branching_matrix=params.branching_matrix,
        branching_ratio=params.branching_ratio,
    )
