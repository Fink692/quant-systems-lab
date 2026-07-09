.PHONY: install test fetch-real-data reproduce-strategy demo-report resume-artifacts

install:
	python -m pip install -e ".[dev]"

test:
	pytest

fetch-real-data:
	python examples/fetch_shiller_sp500_data.py --output data/real/shiller_sp500_monthly.csv

reproduce-strategy:
	python examples/run_valuation_regime_study.py --data data/real/shiller_sp500_monthly.csv --config config/valuation_regime.json --output reports/valuation_regime_study.md

demo-report:
	quantlab demo-report --seed 7 --output examples/demo_report_seed7.md

resume-artifacts:
	python examples/generate_resume_artifacts.py --seed 7
