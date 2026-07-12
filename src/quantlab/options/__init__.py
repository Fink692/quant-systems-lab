"""Option pricing, volatility models, and surface diagnostics."""

from quantlab.options.black_scholes import black_scholes_price, implied_volatility
from quantlab.options.calibration import (
    CalibrationResult,
    OptionQuote,
    calibrate_bates,
    calibrate_heston,
    calibrate_sabr_smile,
)
from quantlab.options.density import RiskNeutralDensity, breeden_litzenberger_density, density_from_price_surface
from quantlab.options.diagnostics import PricingErrorReport, evaluate_option_pricer, pricing_error_report
from quantlab.options.greeks import BlackScholesGreeks, black_scholes_greeks
from quantlab.options.hedging import DeltaHedgeResult, simulate_delta_hedge
from quantlab.options.heston import HestonParams, heston_price
from quantlab.options.local_volatility import dupire_local_volatility
from quantlab.options.monte_carlo import black_scholes_monte_carlo_price, heston_monte_carlo_price
from quantlab.options.pde import black_scholes_finite_difference_price
from quantlab.options.portfolio import (
    OptionBookStressResult,
    OptionPosition,
    option_book_greeks,
    option_book_value,
    stress_option_book,
)
from quantlab.options.sabr import SABRParams, sabr_implied_volatility, sabr_price
from quantlab.options.sabr_surface import (
    SABRSurfaceCalibrationResult,
    calibrate_sabr_surface,
    sabr_surface_implied_volatility,
)
from quantlab.options.ssvi import (
    SSVIArbitrageCheck,
    SSVIParams,
    check_ssvi_no_arbitrage,
    ssvi_implied_volatility,
    ssvi_surface,
    ssvi_total_variance,
)
from quantlab.options.surface import VolatilitySurface, build_volatility_surface_from_chain
from quantlab.options.surface_arbitrage import (
    ArbitrageViolation,
    detect_butterfly_arbitrage,
    detect_calendar_arbitrage,
    detect_call_price_bounds,
    detect_surface_arbitrage,
    detect_vertical_spread_arbitrage,
)
from quantlab.options.surface_repair import SurfaceRepairResult, repair_call_price_surface
from quantlab.options.surface_stability import SurfaceStabilityReport, diagnose_surface_interpolation_stability
from quantlab.options.svi import (
    SVICalibrationResult,
    SVIParams,
    calibrate_svi_slice,
    svi_implied_volatility,
    svi_total_variance,
)
from quantlab.options.variance_reduction import (
    MonteCarloEstimate,
    black_scholes_antithetic_price,
    black_scholes_control_variate_price,
)

__all__ = [
    "CalibrationResult",
    "ArbitrageViolation",
    "BlackScholesGreeks",
    "DeltaHedgeResult",
    "HestonParams",
    "MonteCarloEstimate",
    "OptionQuote",
    "OptionBookStressResult",
    "OptionPosition",
    "PricingErrorReport",
    "RiskNeutralDensity",
    "SABRParams",
    "SABRSurfaceCalibrationResult",
    "SSVIArbitrageCheck",
    "SSVIParams",
    "SVICalibrationResult",
    "SVIParams",
    "SurfaceRepairResult",
    "SurfaceStabilityReport",
    "VolatilitySurface",
    "black_scholes_finite_difference_price",
    "black_scholes_antithetic_price",
    "black_scholes_control_variate_price",
    "black_scholes_greeks",
    "black_scholes_monte_carlo_price",
    "black_scholes_price",
    "build_volatility_surface_from_chain",
    "breeden_litzenberger_density",
    "calibrate_bates",
    "calibrate_heston",
    "calibrate_sabr_smile",
    "calibrate_sabr_surface",
    "calibrate_svi_slice",
    "check_ssvi_no_arbitrage",
    "dupire_local_volatility",
    "density_from_price_surface",
    "detect_butterfly_arbitrage",
    "detect_calendar_arbitrage",
    "detect_call_price_bounds",
    "detect_surface_arbitrage",
    "detect_vertical_spread_arbitrage",
    "diagnose_surface_interpolation_stability",
    "evaluate_option_pricer",
    "heston_monte_carlo_price",
    "heston_price",
    "implied_volatility",
    "option_book_greeks",
    "option_book_value",
    "pricing_error_report",
    "repair_call_price_surface",
    "sabr_implied_volatility",
    "sabr_price",
    "sabr_surface_implied_volatility",
    "simulate_delta_hedge",
    "stress_option_book",
    "ssvi_implied_volatility",
    "ssvi_surface",
    "ssvi_total_variance",
    "svi_implied_volatility",
    "svi_total_variance",
]
