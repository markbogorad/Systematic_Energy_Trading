from streamlit_app.utilities.metrics import compute_sharpe_ratio
from streamlit_app.utilities.signals import get_signal
import itertools
import numpy as np

# --- Display section ---
st.markdown("Strategy Outputs")

for strategy in selected_strategies:
    st.markdown(f"{strategy}")

    signal = get_signal(strategy, df.copy(), **params[strategy])
    price_series = df[params[strategy].get("price_col", df.columns[0])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=price_series, name="Price", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=signal * price_series.std() + price_series.mean(),
                             name="Signal", line=dict(color='red', dash='dot')))
    fig.update_layout(title=f"{strategy} Signal on {commodity}", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- Optimization button ---
    with st.expander(f"🔍 Optimize {strategy}"):
        if st.button(f"Run Optimization for {strategy}"):
            st.info("Running optimization...")

            # Define search space per strategy
            search_space = {}
            base_params = params[strategy].copy()

            if strategy in ["Basic Momentum", "Value", "Time-Weighted Momentum"]:
                search_space = {
                    "window": list(range(10, 60, 5)),
                    "threshold": [round(x, 2) for x in np.linspace(0.05, 0.5, 5)]
                }
            elif strategy == "Crossover Momentum":
                search_space = {
                    "fast": [5, 10, 15],
                    "slow": [20, 30, 40],
                    "threshold": [0.05, 0.1, 0.2]
                }
            elif strategy == "Carry":
                search_space = {
                    "threshold": [0.01, 0.05, 0.1, 0.2]
                }

            result = optimize_strategy(strategy, df.copy(), base_params, search_space)
            st.success(f"Best Sharpe: {result['sharpe']:.3f}")
            st.json(result['best_params'])

    st.markdown("---")
