from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.options.calibration import OptionQuote


@dataclass(frozen=True)
class PricingErrorReport:
    residuals: pd.DataFrame
    rmse: float
    mae: float
    max_abs_error: float
    weighted_rmse: float


def pricing_error_report(quotes: list[OptionQuote], model_prices: np.ndarray) -> PricingErrorReport:
    """Summarize model-vs-market option pricing errors."""
    if not quotes:
        raise ValueError("quotes cannot be empty")
    model = np.asarray(model_prices, dtype=float)
    if model.shape != (len(quotes),):
        raise ValueError("model_prices must have one value per quote")

    market = np.array([quote.price for quote in quotes], dtype=float)
    weights = np.array([quote.weight for quote in quotes], dtype=float)
    if np.any(weights < 0):
        raise ValueError("quote weights must be non-negative")
    errors = model - market
    residuals = pd.DataFrame(
        {
            "strike": [quote.strike for quote in quotes],
            "maturity": [quote.maturity for quote in quotes],
            "option_type": [quote.option_type for quote in quotes],
            "market_price": market,
            "model_price": model,
            "error": errors,
            "abs_error": np.abs(errors),
            "weight": weights,
        }
    )
    weighted_denominator = weights.sum()
    weighted_mse = np.mean(errors**2) if weighted_denominator == 0 else np.sum(weights * errors**2) / weighted_denominator
    return PricingErrorReport(
        residuals=residuals,
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        max_abs_error=float(np.max(np.abs(errors))),
        weighted_rmse=float(np.sqrt(weighted_mse)),
    )


def evaluate_option_pricer(
    quotes: list[OptionQuote],
    pricer: Callable[[OptionQuote], float],
) -> PricingErrorReport:
    """Evaluate a quote-level pricer against a calibration basket."""
    model_prices = np.array([pricer(quote) for quote in quotes], dtype=float)
    return pricing_error_report(quotes, model_prices)
