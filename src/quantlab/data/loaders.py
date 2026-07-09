from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataValidationResult:
    rows: int
    columns: int
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0


def validate_option_chain(option_chain: pd.DataFrame) -> DataValidationResult:
    """Validate the canonical option-chain schema used by the lab."""
    required = {"spot", "strike", "maturity", "rate", "dividend", "option_type", "implied_volatility", "price"}
    issues: list[str] = []
    missing = required - set(option_chain.columns)
    if missing:
        issues.append(f"missing columns: {sorted(missing)}")
        return _result(option_chain, issues)
    if option_chain.empty:
        issues.append("option chain is empty")
    if option_chain[list(required)].isna().any().any():
        issues.append("required columns contain missing values")
    if not set(option_chain["option_type"].astype(str)).issubset({"call", "put"}):
        issues.append("option_type must contain only call/put")
    for column in ("spot", "strike", "maturity", "implied_volatility"):
        if (option_chain[column].astype(float) <= 0).any():
            issues.append(f"{column} must be positive")
    if (option_chain["price"].astype(float) < 0).any():
        issues.append("price must be non-negative")
    if option_chain.duplicated(["maturity", "strike", "option_type"]).any():
        issues.append("duplicate maturity/strike/option_type rows")
    return _result(option_chain, issues)


def load_option_chain_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate an option-chain CSV."""
    frame = pd.read_csv(path)
    result = validate_option_chain(frame)
    if not result.is_valid:
        raise ValueError("; ".join(result.issues))
    return frame


def validate_price_panel(prices: pd.DataFrame) -> DataValidationResult:
    """Validate a positive numeric asset-price panel."""
    issues: list[str] = []
    if prices.empty:
        issues.append("price panel is empty")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        issues.append("price panel must be numeric and complete")
    if (numeric <= 0).any().any():
        issues.append("prices must be positive")
    if prices.columns.duplicated().any():
        issues.append("duplicate asset columns")
    return _result(prices, issues)


def load_price_panel_csv(path: str | Path, date_column: str | None = None) -> pd.DataFrame:
    """Load a wide price panel from CSV and validate it."""
    frame = pd.read_csv(path)
    if date_column is not None:
        if date_column not in frame.columns:
            raise ValueError(f"date_column not found: {date_column}")
        frame[date_column] = pd.to_datetime(frame[date_column])
        frame = frame.set_index(date_column)
    result = validate_price_panel(frame)
    if not result.is_valid:
        raise ValueError("; ".join(result.issues))
    return frame.astype(float)


def returns_from_prices(prices: pd.DataFrame, method: str = "log", dropna: bool = True) -> pd.DataFrame:
    """Convert a positive price panel into simple or log returns."""
    validation = validate_price_panel(prices)
    if not validation.is_valid:
        raise ValueError("; ".join(validation.issues))
    if method not in {"log", "simple"}:
        raise ValueError("method must be 'log' or 'simple'")
    numeric = prices.astype(float)
    returns = np.log(numeric / numeric.shift(1)) if method == "log" else numeric.pct_change()
    return returns.dropna(how="all") if dropna else returns


def validate_credit_spread_curve(curve: pd.DataFrame) -> DataValidationResult:
    """Validate a maturity/credit_spread curve for bootstrapping."""
    required = {"maturity", "credit_spread"}
    issues: list[str] = []
    missing = required - set(curve.columns)
    if missing:
        issues.append(f"missing columns: {sorted(missing)}")
        return _result(curve, issues)
    if curve.empty:
        issues.append("credit spread curve is empty")
    maturities = pd.to_numeric(curve["maturity"], errors="coerce")
    spreads = pd.to_numeric(curve["credit_spread"], errors="coerce")
    if maturities.isna().any() or spreads.isna().any():
        issues.append("maturity and credit_spread must be numeric")
    if (maturities <= 0).any():
        issues.append("maturity must be positive")
    if not maturities.is_monotonic_increasing or maturities.duplicated().any():
        issues.append("maturity must be strictly increasing")
    if (spreads < 0).any():
        issues.append("credit_spread must be non-negative")
    return _result(curve, issues)


def load_credit_spread_curve_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a maturity/credit_spread CSV."""
    frame = pd.read_csv(path)
    result = validate_credit_spread_curve(frame)
    if not result.is_valid:
        raise ValueError("; ".join(result.issues))
    return frame.astype({"maturity": float, "credit_spread": float})


def _result(frame: pd.DataFrame, issues: list[str]) -> DataValidationResult:
    return DataValidationResult(rows=int(len(frame)), columns=int(frame.shape[1]), issues=tuple(issues))
