import numpy as np

from quantlab.market_making.toxicity import adverse_selection_report, order_flow_imbalance, volume_synchronized_pin
from quantlab.portfolio.constraints import apply_weight_bounds, turnover_constrained_mean_variance_weights
from quantlab.rl.q_learning import train_tabular_q_learning
from quantlab.systemic.debtrank import debt_rank
from quantlab.workflows.demo_suite import run_full_demo


def test_order_flow_toxicity_metrics():
    buy = np.array([10.0, 5.0, 0.0])
    sell = np.array([0.0, 5.0, 10.0])
    imbalance = order_flow_imbalance(buy, sell)
    assert np.allclose(imbalance, np.array([1.0, 0.0, -1.0]))

    signed_volume = np.array([3.0, 4.0, -2.0, -6.0, 5.0])
    vpin = volume_synchronized_pin(signed_volume, bucket_volume=5.0)
    assert len(vpin) >= 2
    assert np.all((vpin >= 0.0) & (vpin <= 1.0))

    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    prices = np.array([100.0, 101.0, 100.5, 101.5, 101.0])
    report = adverse_selection_report(signs, prices, horizon=1)
    assert report.observations == 4
    assert 0.0 <= report.hit_rate <= 1.0
    assert set(report.by_sign.index) == {-1.0, 1.0}


def test_tabular_q_learning_returns_valid_policy_table():
    prices = np.linspace(100.0, 120.0, 40)
    result = train_tabular_q_learning(prices, candidate_weights=np.array([-1.0, 0.0, 1.0]), episodes=20, epsilon=0.1, seed=5)
    assert result.q_table.shape == (3, 3)
    assert len(result.episode_rewards) == 20
    assert np.isfinite(result.q_table).all()
    assert result.policy(type("State", (), {"price": 120.0})(), previous_price=119.0) in {-1.0, 0.0, 1.0}


def test_turnover_constrained_optimizer_respects_turnover_limit():
    mu = np.array([0.08, 0.04, 0.02])
    cov = np.diag([0.10, 0.05, 0.03])
    previous = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    weights = turnover_constrained_mean_variance_weights(mu, cov, previous, max_turnover=0.2, risk_aversion=2.0)
    assert abs(weights.sum() - 1.0) < 1e-8
    assert np.sum(np.abs(weights - previous)) <= 0.200001
    assert np.all(weights >= -1e-10)

    bounded = apply_weight_bounds(np.array([-0.2, 0.5, 1.5]), lower=0.0, upper=1.0)
    assert abs(bounded.sum() - 1.0) < 1e-12
    assert np.all((bounded >= 0.0) & (bounded <= 1.0))


def test_debtrank_propagates_initial_distress():
    exposures = np.array([[0.0, 50.0, 0.0], [0.0, 0.0, 40.0], [0.0, 0.0, 0.0]])
    capital = np.array([100.0, 100.0, 100.0])
    result = debt_rank(exposures, capital, initial_distress=np.array([0.0, 0.0, 1.0]), damping=0.8)
    assert result.rounds >= 1
    assert result.distress[2] == 1.0
    assert result.total_impact > 0.0
    assert np.all((result.distress >= 0.0) & (result.distress <= 1.0))


def test_full_demo_exposes_agent_execution_constraints_and_debtrank():
    result = run_full_demo(seed=12).as_dict()
    assert "average_signed_move" in result["market_making"]
    assert "mean_vpin" in result["market_making"]
    assert "q_learning_last_reward" in result["rl_trading"]
    assert "turnover_first_weight" in result["portfolio"]
    assert "debt_rank_impact" in result["systemic"]
