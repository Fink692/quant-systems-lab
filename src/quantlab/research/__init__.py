"""Research workflows for real-data strategy studies."""

from quantlab.research.experiment_config import (
    CostModel,
    DatasetReference,
    EvaluationSettings,
    ExecutionSettings,
    MarketMakingExperimentConfig,
    ResearchPeriod,
    StrategySettings,
    load_market_making_experiment_config,
)
from quantlab.research.leveraged_trend import (
    LeveragedTrendConfig,
    LeveragedTrendResult,
    load_leveraged_etf_csv,
    run_leveraged_trend_study,
)
from quantlab.research.leveraged_trend_stress import (
    LeveragedTrendStressConfig,
    LeveragedTrendStressResult,
    load_qqq_fred_csv,
    run_leveraged_trend_stress,
)
from quantlab.research.market_making_study import (
    MarketMakingStudyResult,
    calibrate_market_data,
    chronological_split,
    chronological_split_from_config,
    run_market_making_study,
)
from quantlab.research.registry import register_experiment
from quantlab.research.valuation_regime import (
    RobustnessResult,
    TearSheet,
    ValuationRegimeConfig,
    ValuationRegimeResult,
    load_shiller_sp500_csv,
    run_valuation_regime_walk_forward,
)

__all__ = [
    "CostModel",
    "DatasetReference",
    "EvaluationSettings",
    "ExecutionSettings",
    "MarketMakingExperimentConfig",
    "MarketMakingStudyResult",
    "ResearchPeriod",
    "StrategySettings",
    "LeveragedTrendConfig",
    "LeveragedTrendResult",
    "LeveragedTrendStressConfig",
    "LeveragedTrendStressResult",
    "RobustnessResult",
    "TearSheet",
    "ValuationRegimeConfig",
    "ValuationRegimeResult",
    "load_leveraged_etf_csv",
    "load_qqq_fred_csv",
    "load_shiller_sp500_csv",
    "load_market_making_experiment_config",
    "calibrate_market_data",
    "chronological_split",
    "chronological_split_from_config",
    "register_experiment",
    "run_market_making_study",
    "run_leveraged_trend_study",
    "run_leveraged_trend_stress",
    "run_valuation_regime_walk_forward",
]
