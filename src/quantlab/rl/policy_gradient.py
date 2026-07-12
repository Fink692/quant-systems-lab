from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.rl.trading_env import TradingEnv, TradingState


@dataclass(frozen=True)
class SoftmaxPolicyGradientResult:
    theta: np.ndarray
    candidate_weights: np.ndarray
    episode_rewards: pd.Series
    feature_scale: float

    def action_probabilities(self, state: TradingState, previous_price: float | None = None) -> np.ndarray:
        features = _policy_features(state, previous_price, self.feature_scale)
        return _softmax(features @ self.theta)

    def policy(self, state: TradingState, previous_price: float | None = None) -> float:
        probabilities = self.action_probabilities(state, previous_price)
        return float(self.candidate_weights[int(np.argmax(probabilities))])


@dataclass(frozen=True)
class PolicyGradientRiskConstraints:
    max_drawdown: float = 0.10
    max_turnover: float = 0.50

    def validate(self) -> None:
        if not 0 <= self.max_drawdown < 1:
            raise ValueError("max_drawdown must be in [0, 1)")
        if self.max_turnover < 0:
            raise ValueError("max_turnover must be non-negative")


@dataclass(frozen=True)
class ConstrainedPolicyGradientResult:
    theta: np.ndarray
    candidate_weights: np.ndarray
    episode_rewards: pd.Series
    adjusted_episode_rewards: pd.Series
    constraint_history: pd.DataFrame
    lagrange_multipliers: pd.Series
    feature_scale: float

    def action_probabilities(self, state: TradingState, previous_price: float | None = None) -> np.ndarray:
        features = _policy_features(state, previous_price, self.feature_scale)
        return _softmax(features @ self.theta)

    def policy(self, state: TradingState, previous_price: float | None = None) -> float:
        probabilities = self.action_probabilities(state, previous_price)
        return float(self.candidate_weights[int(np.argmax(probabilities))])


def train_softmax_policy_gradient(
    prices: np.ndarray,
    candidate_weights: np.ndarray | None = None,
    episodes: int = 80,
    learning_rate: float = 0.05,
    discount_factor: float = 0.95,
    transaction_cost_bps: float = 1.0,
    seed: int | None = None,
) -> SoftmaxPolicyGradientResult:
    """Train a compact REINFORCE-style softmax policy over discrete target weights."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or len(prices) < 5 or np.any(prices <= 0):
        raise ValueError("prices must be a positive one-dimensional array with at least five values")
    if min(episodes, learning_rate, discount_factor) <= 0 or discount_factor > 1:
        raise ValueError("invalid policy-gradient hyperparameters")
    actions = np.array([-1.0, 0.0, 1.0]) if candidate_weights is None else np.asarray(candidate_weights, dtype=float)
    if actions.ndim != 1 or len(actions) == 0:
        raise ValueError("candidate_weights must be one-dimensional and non-empty")
    returns = np.diff(np.log(prices))
    feature_scale = max(float(np.std(returns, ddof=1)), 1e-6)
    theta = np.zeros((3, len(actions)), dtype=float)
    rng = np.random.default_rng(seed)
    episode_rewards = []

    for _episode in range(episodes):
        env = TradingEnv(prices, transaction_cost_bps=transaction_cost_bps)
        state = env.reset()
        previous_price: float | None = None
        features_seen: list[np.ndarray] = []
        probabilities_seen: list[np.ndarray] = []
        actions_seen: list[int] = []
        rewards: list[float] = []
        done = False
        while not done:
            features = _policy_features(state, previous_price, feature_scale)
            probabilities = _softmax(features @ theta)
            action_index = int(rng.choice(len(actions), p=probabilities))
            next_state, reward, done, _info = env.step(float(actions[action_index]))
            features_seen.append(features)
            probabilities_seen.append(probabilities)
            actions_seen.append(action_index)
            rewards.append(float(reward))
            previous_price = state.price
            state = next_state

        returns_to_go = _discounted_returns(np.asarray(rewards), discount_factor)
        if len(returns_to_go) > 1 and returns_to_go.std(ddof=1) > 0:
            returns_to_go = (returns_to_go - returns_to_go.mean()) / returns_to_go.std(ddof=1)
        for features, probabilities, action_index, advantage in zip(
            features_seen, probabilities_seen, actions_seen, returns_to_go
        ):
            gradient = -np.outer(features, probabilities)
            gradient[:, action_index] += features
            theta += learning_rate * advantage * gradient
        episode_rewards.append(float(np.sum(rewards)))

    return SoftmaxPolicyGradientResult(theta, actions, pd.Series(episode_rewards, name="episode_reward"), feature_scale)


def train_constrained_policy_gradient(
    prices: np.ndarray,
    candidate_weights: np.ndarray | None = None,
    constraints: PolicyGradientRiskConstraints | None = None,
    episodes: int = 80,
    learning_rate: float = 0.05,
    discount_factor: float = 0.95,
    transaction_cost_bps: float = 1.0,
    penalty_learning_rate: float = 1.0,
    penalty_cap: float = 25.0,
    seed: int | None = None,
) -> ConstrainedPolicyGradientResult:
    """Train a softmax policy with Lagrangian penalties for drawdown and turnover violations."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or len(prices) < 5 or np.any(prices <= 0):
        raise ValueError("prices must be a positive one-dimensional array with at least five values")
    if min(episodes, learning_rate, discount_factor) <= 0 or discount_factor > 1:
        raise ValueError("invalid policy-gradient hyperparameters")
    if penalty_learning_rate < 0 or penalty_cap <= 0:
        raise ValueError("penalty_learning_rate must be non-negative and penalty_cap must be positive")
    risk_constraints = PolicyGradientRiskConstraints() if constraints is None else constraints
    risk_constraints.validate()
    actions = np.array([-1.0, 0.0, 1.0]) if candidate_weights is None else np.asarray(candidate_weights, dtype=float)
    if actions.ndim != 1 or len(actions) == 0 or np.any(~np.isfinite(actions)):
        raise ValueError("candidate_weights must be one-dimensional, finite, and non-empty")

    returns = np.diff(np.log(prices))
    feature_scale = max(float(np.std(returns, ddof=1)), 1e-6)
    theta = np.zeros((3, len(actions)), dtype=float)
    rng = np.random.default_rng(seed)
    lagrange = np.zeros(2, dtype=float)
    history: list[dict[str, float]] = []

    for _episode in range(episodes):
        env = TradingEnv(prices, transaction_cost_bps=transaction_cost_bps)
        state = env.reset()
        previous_price: float | None = None
        previous_weight = 0.0
        features_seen: list[np.ndarray] = []
        probabilities_seen: list[np.ndarray] = []
        actions_seen: list[int] = []
        rewards: list[float] = []
        adjusted_rewards: list[float] = []
        drawdowns: list[float] = []
        turnovers: list[float] = []
        drawdown_violations: list[float] = []
        turnover_violations: list[float] = []
        done = False
        while not done:
            features = _policy_features(state, previous_price, feature_scale)
            probabilities = _softmax(features @ theta)
            action_index = int(rng.choice(len(actions), p=probabilities))
            action_weight = float(actions[action_index])
            next_state, reward, done, info = env.step(action_weight)
            turnover = abs(action_weight - previous_weight)
            drawdown = max(float(info["drawdown"]), 0.0)
            drawdown_violation = max(drawdown - risk_constraints.max_drawdown, 0.0)
            turnover_violation = max(turnover - risk_constraints.max_turnover, 0.0)
            penalty = lagrange[0] * drawdown_violation + lagrange[1] * turnover_violation

            features_seen.append(features)
            probabilities_seen.append(probabilities)
            actions_seen.append(action_index)
            rewards.append(float(reward))
            adjusted_rewards.append(float(reward - penalty))
            drawdowns.append(drawdown)
            turnovers.append(turnover)
            drawdown_violations.append(drawdown_violation)
            turnover_violations.append(turnover_violation)

            previous_price = state.price
            previous_weight = action_weight
            state = next_state

        returns_to_go = _discounted_returns(np.asarray(adjusted_rewards), discount_factor)
        if len(returns_to_go) > 1 and returns_to_go.std(ddof=1) > 0:
            returns_to_go = (returns_to_go - returns_to_go.mean()) / returns_to_go.std(ddof=1)
        for features, probabilities, action_index, advantage in zip(
            features_seen, probabilities_seen, actions_seen, returns_to_go
        ):
            gradient = -np.outer(features, probabilities)
            gradient[:, action_index] += features
            theta += learning_rate * advantage * gradient

        episode_drawdown_violation = float(max(drawdown_violations, default=0.0))
        episode_turnover_violation = float(np.mean(turnover_violations)) if turnover_violations else 0.0
        lagrange = np.clip(
            lagrange + penalty_learning_rate * np.array([episode_drawdown_violation, episode_turnover_violation]),
            0.0,
            penalty_cap,
        )
        history.append(
            {
                "reward": float(np.sum(rewards)),
                "adjusted_reward": float(np.sum(adjusted_rewards)),
                "max_drawdown": float(max(drawdowns, default=0.0)),
                "average_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
                "drawdown_violation": episode_drawdown_violation,
                "turnover_violation": episode_turnover_violation,
                "lambda_drawdown": float(lagrange[0]),
                "lambda_turnover": float(lagrange[1]),
            }
        )

    constraint_history = pd.DataFrame(history)
    return ConstrainedPolicyGradientResult(
        theta=theta,
        candidate_weights=actions,
        episode_rewards=constraint_history["reward"].rename("episode_reward"),
        adjusted_episode_rewards=constraint_history["adjusted_reward"].rename("adjusted_episode_reward"),
        constraint_history=constraint_history,
        lagrange_multipliers=pd.Series({"drawdown": float(lagrange[0]), "turnover": float(lagrange[1])}),
        feature_scale=feature_scale,
    )


def _policy_features(state: TradingState, previous_price: float | None, feature_scale: float) -> np.ndarray:
    if previous_price is None or previous_price <= 0:
        momentum = 0.0
    else:
        momentum = (state.price / previous_price - 1.0) / feature_scale
    equity_drawdown = 0.0 if state.peak_equity <= 0 else max(1.0 - state.equity / state.peak_equity, 0.0)
    return np.array([1.0, momentum, equity_drawdown], dtype=float)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def _discounted_returns(rewards: np.ndarray, discount_factor: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=float)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + discount_factor * running
        out[idx] = running
    return out
