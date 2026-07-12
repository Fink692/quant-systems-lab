.PHONY: install test quality docs audit fetch-real-data reproduce-strategy fetch-order-book-data reproduce-market-making-sample market-making-dashboard market-making-notebook market-making-paper market-making-video demo-report resume-artifacts

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

demo-report:
	quantlab demo-report --seed 7 --output examples/demo_report_seed7.md

resume-artifacts:
	python examples/generate_resume_artifacts.py --seed 7
