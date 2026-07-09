from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.risk.factor_model import FactorModelResult, fit_factor_model


@dataclass(frozen=True)
class RollingFactorValidationResult:
    report: pd.DataFrame

    @property
    def mean_oos_r_squared(self) -> float:
        return float(self.report["oos_r_squared"].mean())

    @property
    def mean_abs_residual_correlation(self) -> float:
        return float(self.report["mean_abs_residual_correlation"].mean())

    @property
    def mean_specific_risk_share(self) -> float:
        return float(self.report["specific_risk_share"].mean())

    @property
    def max_covariance_condition_number(self) -> float:
        return float(self.report["covariance_condition_number"].max())


def rolling_factor_model_validation(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    train_window: int,
    test_window: int = 20,
    step_size: int | None = None,
) -> RollingFactorValidationResult:
    """Validate a factor model with rolling out-of-sample prediction diagnostics."""
    aligned_assets, aligned_factors = asset_returns.align(factor_returns, join="inner", axis=0)
    if aligned_assets.empty or aligned_factors.empty:
        raise ValueError("asset_returns and factor_returns must overlap on index")
    if train_window <= aligned_factors.shape[1] + 1:
        raise ValueError("train_window must exceed the number of fitted coefficients")
    if test_window <= 0:
        raise ValueError("test_window must be positive")
    step = test_window if step_size is None else step_size
    if step <= 0:
        raise ValueError("step_size must be positive")
    if train_window + test_window > len(aligned_assets):
        raise ValueError("not enough observations for requested train/test windows")

    rows: list[dict[str, float | pd.Timestamp]] = []
    start = 0
    while start + train_window + test_window <= len(aligned_assets):
        train_slice = slice(start, start + train_window)
        test_slice = slice(start + train_window, start + train_window + test_window)
        train_assets = aligned_assets.iloc[train_slice]
        train_factors = aligned_factors.iloc[train_slice]
        test_assets = aligned_assets.iloc[test_slice]
        test_factors = aligned_factors.iloc[test_slice]

        model = fit_factor_model(train_assets, train_factors)
        predictions = _predict_factor_returns(model, test_factors)
        residuals = test_assets - predictions
        rows.append(
            {
                "train_start": train_assets.index[0],
                "train_end": train_assets.index[-1],
                "test_start": test_assets.index[0],
                "test_end": test_assets.index[-1],
                "oos_r_squared": _out_of_sample_r_squared(test_assets, predictions, train_assets.mean(axis=0)),
                "mean_abs_residual_correlation": _mean_abs_off_diagonal_correlation(residuals),
                "specific_risk_share": _specific_risk_share(model),
                "covariance_condition_number": _covariance_condition_number(model.covariance_matrix()),
            }
        )
        start += step

    return RollingFactorValidationResult(pd.DataFrame(rows))


def _predict_factor_returns(model: FactorModelResult, factor_returns: pd.DataFrame) -> pd.DataFrame:
    factors = factor_returns.reindex(columns=model.exposures.columns).astype(float)
    if factors.isna().any().any():
        raise ValueError("factor_returns must cover fitted factor columns")
    predicted = model.intercepts.to_numpy() + factors.to_numpy() @ model.exposures.to_numpy().T
    return pd.DataFrame(predicted, index=factor_returns.index, columns=model.exposures.index)


def _out_of_sample_r_squared(actual: pd.DataFrame, predicted: pd.DataFrame, benchmark_mean: pd.Series) -> float:
    benchmark = benchmark_mean.reindex(actual.columns).astype(float)
    errors = actual.to_numpy() - predicted.reindex(columns=actual.columns).to_numpy()
    centered = actual.to_numpy() - benchmark.to_numpy()[None, :]
    sse = float(np.sum(errors**2))
    sst = float(np.sum(centered**2))
    return 0.0 if sst <= 0 else float(1.0 - sse / sst)


def _mean_abs_off_diagonal_correlation(residuals: pd.DataFrame) -> float:
    if residuals.shape[0] < 2 or residuals.shape[1] < 2:
        return 0.0
    corr = residuals.corr().to_numpy(dtype=float)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = corr[mask]
    finite = values[np.isfinite(values)]
    return 0.0 if len(finite) == 0 else float(np.mean(np.abs(finite)))


def _specific_risk_share(model: FactorModelResult) -> float:
    covariance = model.covariance_matrix().to_numpy(dtype=float)
    total_variance = float(np.trace(covariance))
    if total_variance <= 0:
        return 0.0
    return float(np.sum(model.specific_variance.to_numpy(dtype=float)) / total_variance)


def _covariance_condition_number(covariance: pd.DataFrame) -> float:
    matrix = covariance.to_numpy(dtype=float)
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    positive = eigenvalues[eigenvalues > 1e-12]
    if len(positive) == 0:
        return float("inf")
    return float(max(np.max(eigenvalues), 0.0) / np.min(positive))
