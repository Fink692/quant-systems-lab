# Case Study: Queue-Aware Market Making

This case study shows how Quant Systems Lab turns a theoretical Avellaneda-Stoikov quoting model into a more realistic book-level simulation with queue position, market-order flow, cancellations, inventory, and mark-to-market PnL.

![Queue-aware market-making PnL and inventory](https://raw.githubusercontent.com/Fink692/quant-systems-lab/main/examples/artifacts/market_making_pnl_inventory.svg)

## Problem

A simple market-making simulator often assumes that quoted bid/ask orders fill with an abstract probability. That is useful for first-pass control logic, but it hides the core market-microstructure question: where is the agent in the queue, and how much incoming market-order flow must trade through prior resting depth before the agent gets filled?

The queue-aware simulator addresses this by putting the agent inside a price-level limit order book. Each step:

1. Builds or replenishes visible book depth around the mid price.
2. Computes Avellaneda-Stoikov bid/ask quotes from inventory and volatility.
3. Snaps quotes to the tick grid and records queue-ahead depth.
4. Adds the agent's bid and ask orders behind existing depth.
5. Simulates buy/sell market-order arrivals.
6. Attributes fills to the agent only after queue-ahead depth is consumed.
7. Cancels residual agent quantity, replenishes book depth, and marks inventory to market.

## Why It Matters

This separates two ideas that are often blended together:

- Quoting policy: where the agent wants to quote based on spread capture and inventory risk.
- Execution mechanics: whether the market trades through enough queue depth for the agent to actually fill.

That distinction is the difference between a toy market-maker and a simulator that can discuss adverse selection, latency, inventory, and realized fill quality.

## Model Components

- `quantlab.market_making.avellaneda_stoikov`: reservation price and optimal spread logic.
- `quantlab.market_making.limit_order_book`: price-level depth, market orders, cancellations, and queue depth lookup.
- `quantlab.market_making.book_simulator`: queue-aware agent simulation and fill attribution.
- `quantlab.market_making.hawkes`: clustered buy/sell order-flow simulation.
- `quantlab.market_making.attribution`: spread capture, inventory mark-to-market, and slippage decomposition.

## Reproduce The Visual

```powershell
$env:PYTHONPATH='src'
python examples\generate_resume_artifacts.py --output-dir examples\artifacts --seed 7
```

The generated chart reports the deterministic seed-7 book simulation with positive realized fills, inventory movement, and final PnL.

## Interview Notes

Good follow-up questions to be ready for:

- Why should queue-ahead depth matter more than a pure fill probability?
- How does inventory shift the reservation price in Avellaneda-Stoikov?
- What adverse-selection signal would you add next?
- How would latency change the realized spread capture?
- What data would be needed to calibrate market-order intensity and cancellation rates from a real venue?
