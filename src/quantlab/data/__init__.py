"""Synthetic datasets for examples, tests, and demos."""

from quantlab.data.loaders import (
    ORDER_BOOK_EVENT_SCHEMA_VERSION,
    DataValidationResult,
    load_credit_spread_curve_csv,
    load_option_chain_csv,
    load_order_book_events_csv,
    load_price_panel_csv,
    returns_from_prices,
    validate_credit_spread_curve,
    validate_option_chain,
    validate_order_book_events,
    validate_price_panel,
)
from quantlab.data.synthetic import (
    SyntheticFactorPanel,
    synthetic_cointegrated_prices,
    synthetic_credit_spreads,
    synthetic_exposure_network,
    synthetic_factor_panel,
    synthetic_option_chain,
)

__all__ = [
    "DataValidationResult",
    "ORDER_BOOK_EVENT_SCHEMA_VERSION",
    "SyntheticFactorPanel",
    "load_credit_spread_curve_csv",
    "load_option_chain_csv",
    "load_order_book_events_csv",
    "load_price_panel_csv",
    "returns_from_prices",
    "synthetic_cointegrated_prices",
    "synthetic_credit_spreads",
    "synthetic_exposure_network",
    "synthetic_factor_panel",
    "synthetic_option_chain",
    "validate_credit_spread_curve",
    "validate_option_chain",
    "validate_order_book_events",
    "validate_price_panel",
]
