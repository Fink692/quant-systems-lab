from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_transition_matrix(matrix: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Normalize rows of a rating transition matrix."""
    is_frame = isinstance(matrix, pd.DataFrame)
    values = matrix.to_numpy(dtype=float) if is_frame else np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or np.any(values < 0):
        raise ValueError("transition matrix must be square and non-negative")
    row_sums = values.sum(axis=1)
    if np.any(row_sums == 0):
        raise ValueError("transition matrix rows must have positive sums")
    normalized = values / row_sums[:, None]
    if is_frame:
        return pd.DataFrame(normalized, index=matrix.index, columns=matrix.columns)
    return normalized


def transition_matrix_power(matrix: pd.DataFrame | np.ndarray, periods: int) -> pd.DataFrame | np.ndarray:
    """Compute multi-period transition probabilities."""
    if periods < 1:
        raise ValueError("periods must be positive")
    normalized = normalize_transition_matrix(matrix)
    is_frame = isinstance(normalized, pd.DataFrame)
    values = normalized.to_numpy(dtype=float) if is_frame else normalized
    powered = np.linalg.matrix_power(values, periods)
    if is_frame:
        return pd.DataFrame(powered, index=normalized.index, columns=normalized.columns)
    return powered


def cumulative_default_probability(
    matrix: pd.DataFrame | np.ndarray,
    start_rating: str | int,
    periods: int,
    default_state: str | int = "D",
) -> float:
    """Return cumulative probability of default by horizon from a transition matrix."""
    powered = transition_matrix_power(matrix, periods)
    if isinstance(powered, pd.DataFrame):
        return float(powered.loc[start_rating, default_state])
    return float(powered[int(start_rating), int(default_state)])


def simulate_rating_paths(
    matrix: pd.DataFrame | np.ndarray,
    start_rating: str | int,
    periods: int,
    paths: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate Markov-chain rating paths."""
    if periods < 1 or paths < 1:
        raise ValueError("periods and paths must be positive")
    normalized = normalize_transition_matrix(matrix)
    is_frame = isinstance(normalized, pd.DataFrame)
    states = list(normalized.index) if is_frame else list(range(normalized.shape[0]))
    probabilities = normalized.to_numpy(dtype=float) if is_frame else normalized
    start_index = states.index(start_rating) if is_frame else int(start_rating)
    rng = np.random.default_rng(seed)
    results = np.empty((paths, periods + 1), dtype=object)
    results[:, 0] = states[start_index]
    current = np.full(paths, start_index, dtype=int)
    for period in range(1, periods + 1):
        for path_idx in range(paths):
            current[path_idx] = int(rng.choice(len(states), p=probabilities[current[path_idx]]))
            results[path_idx, period] = states[current[path_idx]]
    return pd.DataFrame(results, columns=[f"t{idx}" for idx in range(periods + 1)])
