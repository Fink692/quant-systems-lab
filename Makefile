.PHONY: install test quality docs audit fetch-real-data fetch-leveraged-data fetch-long-history-data fetch-execution-ohlc reproduce-strategy reproduce-leveraged-strategy reproduce-long-history-stress reproduce-execution-audit record-paper-decision score-paper-outcome fetch-order-book-data reproduce-market-making-sample market-making-dashboard market-making-notebook market-making-paper market-making-video demo-report resume-artifacts

install:
	python -m pip install -e ".[dev]"

test:
	pytest --cov=quantlab --cov-report=term-missing --cov-fail-under=85

quality:
	ruff check src tests examples
	black --check src tests examples
	mypy

docs:
	mkdocs build --strict

audit:
	pip-audit

fetch-real-data:
	python examples/fetch_shiller_sp500_data.py --output data/real/shiller_sp500_monthly.csv

fetch-leveraged-data:
	python examples/fetch_leveraged_etf_data.py --output data/real/leveraged_etf_adjusted.csv --metadata data/real/leveraged_etf_adjusted.metadata.json

fetch-long-history-data:
	python examples/fetch_qqq_fred_stress_data.py --output data/real/qqq_fred_stress_daily.csv --metadata data/real/qqq_fred_stress_daily.metadata.json

reproduce-strategy:
	python examples/run_valuation_regime_study.py --data data/real/shiller_sp500_monthly.csv --config config/valuation_regime.json --output reports/valuation_regime_study.md

fetch-order-book-data:
	python examples/fetch_lobster_sample.py --output-dir data/real/lobster_sample

reproduce-market-making-sample:
	python examples/run_market_making_flagship.py --data-dir data/real/lobster_sample --output-dir reports/market_making_sample

market-making-dashboard:
	streamlit run examples/market_making_dashboard.py

market-making-notebook:
	jupyter nbconvert --execute --to notebook --inplace notebooks/start_here_market_making.ipynb

market-making-paper:
	python scripts/generate_market_making_paper.py --report-dir reports/market_making_sample --output-dir output/pdf

market-making-video:
	python scripts/generate_market_making_demo_video.py --report-dir reports/market_making_sample --output output/video/queue_aware_market_making_demo.mp4

reproduce-leveraged-strategy:
	python examples/run_leveraged_trend_study.py --data data/real/leveraged_etf_adjusted.csv --config config/leveraged_trend.json --output reports/leveraged_trend_study.md

reproduce-long-history-stress:
	python examples/run_leveraged_trend_stress.py --data data/real/qqq_fred_stress_daily.csv --actual data/real/leveraged_etf_adjusted.csv --config config/leveraged_trend_stress.json --output reports/leveraged_trend_long_history.md

record-paper-decision:
	python examples/record_leveraged_trend_paper.py --snapshot "$(SNAPSHOT)" --metadata "$(METADATA)" --ledger paper/leveraged_trend_decisions.jsonl --effective-session "$(EFFECTIVE_SESSION)" --config config/leveraged_trend_paper.json

fetch-execution-ohlc:
	python examples/fetch_leveraged_etf_ohlc.py --output data/paper/execution_timing_ohlc_2026-07-13.csv --metadata data/paper/execution_timing_ohlc_2026-07-13.metadata.json

reproduce-execution-audit:
	python examples/run_execution_timing_audit.py --data data/paper/execution_timing_ohlc_2026-07-13.csv --output reports/leveraged_trend_execution_timing.md

score-paper-outcome:
	python examples/score_leveraged_trend_paper.py --decisions paper/leveraged_trend_decisions.jsonl --outcomes paper/leveraged_trend_outcomes.jsonl --total-cost-bps 10

demo-report:
	quantlab demo-report --seed 7 --output examples/demo_report_seed7.md

resume-artifacts:
	python examples/generate_resume_artifacts.py --seed 7
