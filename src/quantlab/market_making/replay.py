from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.market_making.strategies import MarketState, QuotingPolicy


@dataclass(frozen=True)
class ReplayConfig:
    tick_size: float
    order_size: float = 10.0
    inventory_limit: float = 100.0
    maker_fee_bps: float = -0.1
    taker_fee_bps: float = 0.3
    fixed_fee_per_order: float = 0.0
    latency_ns: int = 0
    quote_interval_ns: int = 100_000_000
    queue_ahead_fraction: float = 1.0
    toxicity_window: int = 100
    volatility_window: int = 100
    adverse_horizon_ns: int = 1_000_000_000
    record_every: int = 100

    def validate(self) -> None:
        if min(self.tick_size, self.order_size, self.inventory_limit) <= 0:
            raise ValueError("tick_size, order_size, and inventory_limit must be positive")
        if self.latency_ns < 0 or self.quote_interval_ns < 1 or self.record_every < 1:
            raise ValueError("latency must be non-negative and intervals positive")
        if not 0 <= self.queue_ahead_fraction <= 1:
            raise ValueError("queue_ahead_fraction must lie in [0, 1]")
        if self.toxicity_window < 1 or self.volatility_window < 2:
            raise ValueError("toxicity and volatility windows are too short")


@dataclass(frozen=True)
class ReplayResult:
    strategy: str
    history: pd.DataFrame
    fills: pd.DataFrame
    quotes_submitted: int
    gross_pnl: float
    net_pnl: float
    fees: float
    fill_rate: float
    realized_spread: float
    implementation_shortfall: float
    adverse_selection_cost: float
    max_abs_inventory: float
    inventory_std: float
    max_drawdown: float
    expected_shortfall_95: float
    quote_uptime: float
    message_rate_per_hour: float
    accounting_error: float

    def summary(self) -> dict[str, float | str]:
        return {
            "strategy": self.strategy,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "fees": self.fees,
            "fill_rate": self.fill_rate,
            "realized_spread": self.realized_spread,
            "implementation_shortfall": self.implementation_shortfall,
            "adverse_selection_cost": self.adverse_selection_cost,
            "max_abs_inventory": self.max_abs_inventory,
            "inventory_std": self.inventory_std,
            "max_drawdown": self.max_drawdown,
            "expected_shortfall_95": self.expected_shortfall_95,
            "quote_uptime": self.quote_uptime,
            "message_rate_per_hour": self.message_rate_per_hour,
            "accounting_error": self.accounting_error,
        }


@dataclass
class _Order:
    side: str
    price: float
    remaining: float
    queue_ahead: float
    decision_mid: float
    submitted_ns: int


def replay_market_maker(
    events: pd.DataFrame,
    snapshots: pd.DataFrame,
    policy: QuotingPolicy,
    config: ReplayConfig,
) -> ReplayResult:
    """Replay one quoting policy against synchronized historical L2 messages."""
    config.validate()
    if snapshots.empty:
        raise ValueError("snapshots cannot be empty")
    required = {"source_row", "timestamp_ns", "ask_price_1", "bid_price_1", "ask_quantity_1", "bid_quantity_1"}
    missing = required - set(snapshots.columns)
    if missing:
        raise ValueError(f"snapshots missing columns: {sorted(missing)}")

    timeline = snapshots.reset_index(drop=True)
    timestamps = timeline["timestamp_ns"].to_numpy(dtype=np.int64)
    mids = 0.5 * (timeline["ask_price_1"].to_numpy(float) + timeline["bid_price_1"].to_numpy(float))
    event_lookup = {int(row.source_row): row for row in events.itertuples(index=False)}
    row_to_position = {int(row): position for position, row in enumerate(timeline["source_row"].to_numpy(np.int64))}
    signed_flow = np.zeros(len(timeline), dtype=float)
    for row_number, event in event_lookup.items():
        position = row_to_position.get(row_number)
        if position is not None and event.event_type == "trade":
            signed_flow[position] = float(event.quantity) * (1.0 if event.side == "ask" else -1.0)

    cash = 0.0
    fees = 0.0
    inventory = 0.0
    bid_order: _Order | None = None
    ask_order: _Order | None = None
    quote_count = 0
    active_sides = 0
    last_quote_ns = -(10**30)
    fill_rows: list[dict[str, float | int | str]] = []
    history_rows: list[dict[str, float | int]] = []

    for index in range(1, len(timeline)):
        now = int(timestamps[index])
        source_row = int(timeline.iloc[index]["source_row"])
        event = event_lookup.get(source_row)
        if event is not None and bool(event.applies_to_visible_book):
            bid_order, cash, inventory, fees = _consume_event(
                event, bid_order, cash, inventory, fees, config, mids[index], now, fill_rows
            )
            ask_order, cash, inventory, fees = _consume_event(
                event, ask_order, cash, inventory, fees, config, mids[index], now, fill_rows
            )

        if now - last_quote_ns >= config.quote_interval_ns:
            observed_time = now - config.latency_ns
            observed_index = max(int(np.searchsorted(timestamps, observed_time, side="right") - 1), 0)
            state = _market_state(timeline, mids, signed_flow, observed_index, inventory, config)
            bid, ask = policy.quotes(state, config.tick_size)
            bid_order = _new_order("bid", bid, state.mid, now, timeline.iloc[index], inventory, config)
            ask_order = _new_order("ask", ask, state.mid, now, timeline.iloc[index], inventory, config)
            quote_count += int(bid_order is not None) + int(ask_order is not None)
            last_quote_ns = now
        active_sides += int(bid_order is not None) + int(ask_order is not None)

        if index % config.record_every == 0 or index == len(timeline) - 1:
            history_rows.append(
                {
                    "timestamp_ns": now,
                    "mid": mids[index],
                    "inventory": inventory,
                    "cash": cash,
                    "fees": fees,
                    "pnl": cash + inventory * mids[index],
                }
            )

    final_mid = mids[-1]
    if inventory > 0:
        price = float(timeline.iloc[-1]["bid_price_1"])
        liquidation_quantity = inventory
        notional = liquidation_quantity * price
        liquidation_fee = abs(notional) * config.taker_fee_bps / 10_000.0
        cash += notional - liquidation_fee
        fees += liquidation_fee
        fill_rows.append(
            {
                "timestamp_ns": int(timestamps[-1]),
                "side": "ask",
                "price": price,
                "quantity": liquidation_quantity,
                "decision_mid": final_mid,
                "arrival_mid": final_mid,
                "direction": -1.0,
                "fee": liquidation_fee,
                "liquidation": True,
            }
        )
        inventory = 0.0
    elif inventory < 0:
        price = float(timeline.iloc[-1]["ask_price_1"])
        liquidation_quantity = abs(inventory)
        notional = liquidation_quantity * price
        liquidation_fee = notional * config.taker_fee_bps / 10_000.0
        cash -= notional + liquidation_fee
        fees += liquidation_fee
        fill_rows.append(
            {
                "timestamp_ns": int(timestamps[-1]),
                "side": "bid",
                "price": price,
                "quantity": liquidation_quantity,
                "decision_mid": final_mid,
                "arrival_mid": final_mid,
                "direction": 1.0,
                "fee": liquidation_fee,
                "liquidation": True,
            }
        )
        inventory = 0.0
    net_pnl = cash
    fills = pd.DataFrame(fill_rows)
    history = pd.DataFrame(history_rows)
    if not history.empty:
        history.loc[history.index[-1], ["inventory", "cash", "fees", "pnl"]] = [0.0, cash, fees, cash]
    metrics = _execution_metrics(fills, history, timestamps, mids, config)
    duration_hours = max((timestamps[-1] - timestamps[0]) / 3.6e12, 1e-12)
    ledger_cash = (
        float((-fills["direction"] * fills["price"] * fills["quantity"] - fills["fee"]).sum())
        if not fills.empty
        else 0.0
    )
    accounting_error = float(net_pnl - ledger_cash)
    strategy_fills = fills.loc[~fills.get("liquidation", False).astype(bool)] if not fills.empty else fills
    return ReplayResult(
        strategy=policy.name,
        history=history,
        fills=fills,
        quotes_submitted=quote_count,
        gross_pnl=float(net_pnl + fees),
        net_pnl=float(net_pnl),
        fees=float(fees),
        fill_rate=float(len(strategy_fills) / quote_count) if quote_count else 0.0,
        realized_spread=metrics["realized_spread"],
        implementation_shortfall=metrics["implementation_shortfall"],
        adverse_selection_cost=metrics["adverse_selection_cost"],
        max_abs_inventory=metrics["max_abs_inventory"],
        inventory_std=metrics["inventory_std"],
        max_drawdown=metrics["max_drawdown"],
        expected_shortfall_95=metrics["expected_shortfall_95"],
        quote_uptime=float(active_sides / (2 * max(len(timeline) - 1, 1))),
        message_rate_per_hour=float(quote_count / duration_hours),
        accounting_error=accounting_error,
    )


def _consume_event(event, order, cash, inventory, fees, config, arrival_mid, now, fill_rows):
    if order is None or event.side != order.side or abs(float(event.price) - order.price) > 1e-9:
        return order, cash, inventory, fees
    quantity = float(event.quantity)
    if event.event_type == "cancel":
        order.queue_ahead = max(order.queue_ahead - quantity, 0.0)
        return order, cash, inventory, fees
    if event.event_type != "trade":
        return order, cash, inventory, fees
    behind_queue = max(quantity - order.queue_ahead, 0.0)
    order.queue_ahead = max(order.queue_ahead - quantity, 0.0)
    fill_quantity = min(behind_queue, order.remaining)
    if fill_quantity <= 0:
        return order, cash, inventory, fees
    notional = fill_quantity * order.price
    fee = notional * config.maker_fee_bps / 10_000.0 + config.fixed_fee_per_order
    if order.side == "bid":
        cash -= notional + fee
        inventory += fill_quantity
        direction = 1.0
    else:
        cash += notional - fee
        inventory -= fill_quantity
        direction = -1.0
    fees += fee
    order.remaining -= fill_quantity
    fill_rows.append(
        {
            "timestamp_ns": now,
            "side": order.side,
            "price": order.price,
            "quantity": fill_quantity,
            "decision_mid": order.decision_mid,
            "arrival_mid": arrival_mid,
            "direction": direction,
            "fee": fee,
            "liquidation": False,
        }
    )
    return (None if order.remaining <= 1e-9 else order), cash, inventory, fees


def _new_order(side, price, decision_mid, now, snapshot, inventory, config):
    if price is None or price <= 0:
        return None
    if side == "bid" and inventory + config.order_size > config.inventory_limit:
        return None
    if side == "ask" and inventory - config.order_size < -config.inventory_limit:
        return None
    queue = 0.0
    for level in range(1, 51):
        price_key = f"{side}_price_{level}"
        quantity_key = f"{side}_quantity_{level}"
        if price_key not in snapshot.index:
            break
        if abs(float(snapshot[price_key]) - float(price)) <= 1e-9:
            queue = float(snapshot[quantity_key]) * config.queue_ahead_fraction
            break
    return _Order(side, float(price), config.order_size, queue, decision_mid, now)


def _market_state(timeline, mids, signed_flow, index, inventory, config):
    start_tox = max(index - config.toxicity_window + 1, 0)
    flow = signed_flow[start_tox : index + 1]
    toxicity = float(flow.sum() / np.abs(flow).sum()) if np.abs(flow).sum() > 0 else 0.0
    start_vol = max(index - config.volatility_window + 1, 0)
    changes = np.diff(mids[start_vol : index + 1])
    volatility = float(np.std(changes, ddof=1)) if len(changes) > 1 else config.tick_size
    row = timeline.iloc[index]
    return MarketState(
        timestamp_ns=int(row["timestamp_ns"]),
        best_bid=float(row["bid_price_1"]),
        best_ask=float(row["ask_price_1"]),
        bid_quantity=float(row["bid_quantity_1"]),
        ask_quantity=float(row["ask_quantity_1"]),
        inventory=inventory,
        toxicity=toxicity,
        volatility=max(volatility, config.tick_size / 10.0),
    )


def _execution_metrics(fills, history, timestamps, mids, config):
    defaults = {
        "realized_spread": 0.0,
        "implementation_shortfall": 0.0,
        "adverse_selection_cost": 0.0,
        "max_abs_inventory": 0.0,
        "inventory_std": 0.0,
        "max_drawdown": 0.0,
        "expected_shortfall_95": 0.0,
    }
    strategy_fills = fills.loc[~fills.get("liquidation", False).astype(bool)] if not fills.empty else fills
    if not strategy_fills.empty:
        signed_edge = strategy_fills["direction"] * (strategy_fills["arrival_mid"] - strategy_fills["price"])
        decision_edge = strategy_fills["direction"] * (strategy_fills["decision_mid"] - strategy_fills["price"])
        defaults["realized_spread"] = float(
            (signed_edge * strategy_fills["quantity"]).sum() / strategy_fills["quantity"].sum()
        )
        defaults["implementation_shortfall"] = float(
            ((decision_edge - signed_edge) * strategy_fills["quantity"]).sum() / strategy_fills["quantity"].sum()
        )
        future_indices = np.searchsorted(
            timestamps, strategy_fills["timestamp_ns"].to_numpy(np.int64) + config.adverse_horizon_ns, side="left"
        )
        valid = future_indices < len(mids)
        if valid.any():
            future = mids[future_indices[valid]]
            arrival = strategy_fills.loc[valid, "arrival_mid"].to_numpy(float)
            direction = strategy_fills.loc[valid, "direction"].to_numpy(float)
            quantity = strategy_fills.loc[valid, "quantity"].to_numpy(float)
            defaults["adverse_selection_cost"] = float(np.average(-direction * (future - arrival), weights=quantity))
    if not history.empty:
        inventory = history["inventory"].to_numpy(float)
        pnl = history["pnl"].to_numpy(float)
        defaults["max_abs_inventory"] = float(np.max(np.abs(inventory)))
        defaults["inventory_std"] = float(np.std(inventory))
        defaults["max_drawdown"] = float(np.max(np.maximum.accumulate(pnl) - pnl))
        changes = np.diff(pnl)
        if len(changes):
            cutoff = np.quantile(changes, 0.05)
            defaults["expected_shortfall_95"] = float(-np.mean(changes[changes <= cutoff]))
    return defaults
