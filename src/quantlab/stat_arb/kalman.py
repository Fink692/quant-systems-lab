from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KalmanHedgeResult:
    states: pd.DataFrame
    spread: pd.Series
    state_covariances: np.ndarray


def kalman_dynamic_hedge_ratio(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    transition_covariance: float = 1e-5,
    observation_covariance: float | None = None,
) -> KalmanHedgeResult:
    """Estimate time-varying intercept and hedge ratio for y = alpha + beta * x."""
    y_series = _as_series(y, "y")
    x_series = _as_series(x, "x")
    aligned = pd.concat([y_series, x_series], axis=1).dropna()
    if len(aligned) < 5:
        raise ValueError("at least five aligned observations are required")
    if transition_covariance <= 0:
        raise ValueError("transition_covariance must be positive")

    y_values = aligned.iloc[:, 0].to_numpy(dtype=float)
    x_values = aligned.iloc[:, 1].to_numpy(dtype=float)
    seed_window = min(30, len(aligned))
    design = np.column_stack([np.ones(seed_window), x_values[:seed_window]])
    initial_state = np.linalg.lstsq(design, y_values[:seed_window], rcond=None)[0]
    residuals = y_values[:seed_window] - design @ initial_state
    obs_var = float(np.var(residuals, ddof=1)) if observation_covariance is None else float(observation_covariance)
    obs_var = max(obs_var, 1e-10)

    state = initial_state.astype(float)
    covariance = np.eye(2) * 10.0
    transition = np.eye(2)
    transition_noise = np.eye(2) * transition_covariance
    observation_noise = obs_var
    states = np.zeros((len(aligned), 2))
    covariances = np.zeros((len(aligned), 2, 2))
    spreads = np.zeros(len(aligned))

    for idx, (x_t, y_t) in enumerate(zip(x_values, y_values)):
        predicted_state = transition @ state
        predicted_covariance = transition @ covariance @ transition.T + transition_noise
        observation = np.array([1.0, x_t])
        innovation = y_t - observation @ predicted_state
        innovation_variance = float(observation @ predicted_covariance @ observation.T + observation_noise)
        kalman_gain = predicted_covariance @ observation.T / innovation_variance
        state = predicted_state + kalman_gain * innovation
        covariance = predicted_covariance - np.outer(kalman_gain, observation) @ predicted_covariance
        states[idx] = state
        covariances[idx] = covariance
        spreads[idx] = y_t - observation @ state

    state_frame = pd.DataFrame(states, index=aligned.index, columns=["intercept", "hedge_ratio"])
    return KalmanHedgeResult(
        states=state_frame,
        spread=pd.Series(spreads, index=aligned.index, name="kalman_spread"),
        state_covariances=covariances,
    )


def _as_series(values: pd.Series | np.ndarray, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.rename(name)
    return pd.Series(np.asarray(values, dtype=float), name=name)
