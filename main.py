import math
import streamlit as st
import numpy as np
import pandas as pd

DAYS_IN_YEAR = 252


def get_weights(
    number_of_days: int, decay: float, normalize: bool = True
) -> np.ndarray:
    """Generate weights for a given number of days and decay factor."""
    if decay == 1:
        weights = np.ones(number_of_days)
    else:
        weights = (1 - decay) * decay ** np.arange(number_of_days)

    if normalize:
        weights /= np.sum(weights)
        assert math.isclose(np.sum(weights), 1.0)
    return weights


def main():
    st.set_page_config(page_title="Decay Factors and Weights", page_icon="📊")
    st.title("Decay factors and weights")

    col1, col2 = st.columns(2)
    with col1:
        n_months_A = st.slider(
            "Number of months A",
            min_value=1,
            max_value=36,
            value=18,
            step=1,
        )
        decay_A = st.slider(
            "Decay factor A",
            min_value=0.0,
            max_value=1.0,
            value=0.94,
        )
    with col2:
        n_months_B = st.slider(
            "Number of months B",
            min_value=1,
            max_value=36,
            value=18,
            step=1,
        )
        decay_B = st.slider(
            "Decay factor B",
            min_value=0.0,
            max_value=1.0,
            value=0.97,
        )
    normalize = st.checkbox("Normalize weights", value=True)

    data = {
        "A": get_weights(int(DAYS_IN_YEAR * n_months_A / 12), decay_A, normalize),
        "B": get_weights(int(DAYS_IN_YEAR * n_months_B / 12), decay_B, normalize),
    }
    # Converting each series separately to avoid issue with different lengths
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

    st.text("Weights")
    st.line_chart(df, use_container_width=True, x_label="Observation", y_label="Weight")

    st.text("Cumulative Weights")
    st.line_chart(
        df.cumsum(),
        use_container_width=True,
        x_label="Observation",
        y_label="Cumulative Weight",
    )

    # Calculate cumulative weights by month
    months_df = pd.DataFrame()
    for series_name in ["A", "B"]:
        days_per_month = int(DAYS_IN_YEAR / 12)
        n_months = n_months_A if series_name == "A" else n_months_B
        monthly_indices = [days_per_month * i - 1 for i in range(1, n_months + 1)]
        print(monthly_indices)
        cumulative_weights = (
            df[series_name].cumsum().iloc[monthly_indices].reset_index(drop=True)
        )
        months_df[f"{series_name} Cumulative"] = cumulative_weights

    st.text("Cumulative Weights by Month")
    months_df.index = months_df.index + 1
    months_df.index.name = "Month"
    st.dataframe(months_df)


if __name__ == "__main__":
    main()
