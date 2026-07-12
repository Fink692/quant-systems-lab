from __future__ import annotations

from quantlab.reporting.market_making_dashboard import load_dashboard_data


def main() -> None:
    import plotly.express as px
    import streamlit as st

    st.set_page_config(page_title="Queue-Aware Market Making", page_icon="📈", layout="wide")
    data = load_dashboard_data("reports/market_making_sample")
    summary = data["summary"]
    quality = data["quality"]
    comparison = data["comparison"]
    sensitivity = data["sensitivity"]
    validation = data["validation"]

    st.title("Queue-Aware Market Making")
    st.caption("Real-data pipeline validation with chronological evaluation and explicit execution assumptions")
    st.warning(
        "The public LOBSTER sample validates the research pipeline. One session is not evidence of persistent alpha."
    )

    columns = st.columns(4)
    columns[0].metric("Messages", f"{quality['messages']:,}")
    columns[1].metric("Reconstruction match", f"{quality['reconstruction_match_rate']:.2%}")
    columns[2].metric("Selected queue ahead", f"{summary['selected_queue_ahead_fraction']:.2f}")
    columns[3].metric("Accounting max error", f"{comparison['accounting_error'].abs().max():.2e}")

    left, right = st.columns(2)
    with left:
        st.subheader("Held-out policy comparison")
        figure = px.bar(
            comparison.sort_values("net_pnl"),
            x="net_pnl",
            y="strategy",
            orientation="h",
            color="max_abs_inventory",
            labels={"net_pnl": "Net PnL", "strategy": "Policy", "max_abs_inventory": "Max |inventory|"},
            color_continuous_scale="Blues",
        )
        st.plotly_chart(figure, width="stretch")
    with right:
        st.subheader("Validation-only queue selection")
        figure = px.line(validation, x="queue_ahead_fraction", y="risk_score", markers=True)
        st.plotly_chart(figure, width="stretch")

    st.subheader("Latency, queue, and fee sensitivity")
    selected_strategy = st.selectbox("Policy", sorted(sensitivity["strategy"].unique()), index=2)
    fee_multiplier = st.select_slider(
        "Fee multiplier", options=sorted(sensitivity["fee_multiplier"].unique()), value=1.0
    )
    view = sensitivity.loc[
        sensitivity["strategy"].eq(selected_strategy) & sensitivity["fee_multiplier"].eq(fee_multiplier)
    ].copy()
    view["latency_ms"] = view["latency_ns"] / 1e6
    figure = px.line(
        view,
        x="queue_ahead_fraction",
        y="net_pnl",
        color="latency_ms",
        markers=True,
        labels={"latency_ms": "Latency (ms)", "queue_ahead_fraction": "Queue-ahead fraction", "net_pnl": "Net PnL"},
    )
    st.plotly_chart(figure, width="stretch")

    with st.expander("Audit fingerprints"):
        st.code(
            f"dataset={summary['dataset_fingerprint']}\nconfig={summary['config_fingerprint']}",
            language="text",
        )
    st.dataframe(comparison, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
