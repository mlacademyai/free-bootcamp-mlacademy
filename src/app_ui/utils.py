from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def load_data(path: Path) -> pd.DataFrame | None:
    """Load parquet file and parse datetime column."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def create_figure(
    df_actual: pd.DataFrame, 
    df_pred: pd.DataFrame | None, 
    lookback_hours: int,
    datetime_col: str = "datetime",
) -> go.Figure:
    """Create the predictions vs actual plot."""
    # Calculate time range
    if df_pred is not None and not df_pred.empty:
        max_time = df_pred[datetime_col].max()
        current_time = max_time - pd.Timedelta(hours=1)
    else:
        current_time = df_actual[datetime_col].max()
        max_time = current_time

    min_time = max_time - pd.Timedelta(hours=lookback_hours)

    fig = go.Figure()

    # Add predictions trace
    if df_pred is not None:
        df_pred_f = df_pred[(df_pred[datetime_col] >= min_time) & (df_pred[datetime_col] <= max_time)]
        if not df_pred_f.empty:
            fig.add_trace(go.Scattergl(
                x=df_pred_f[datetime_col], y=df_pred_f["prediction"],
                name="Predicted", mode="lines+markers",
                line=dict(color="#1E8449", width=2), marker=dict(size=8),
            ))

    # Add actual trace
    df_actual_f = df_actual[(df_actual[datetime_col] >= min_time) & (df_actual[datetime_col] <= current_time)]
    fig.add_trace(go.Scattergl(
        x=df_actual_f[datetime_col], y=df_actual_f["cnt"],
        name="Actual", mode="lines+markers",
        line=dict(color="#F08080", width=2), marker=dict(symbol="x", size=6),
    ))

    # Add vertical line at end of actual data
    if not df_actual_f.empty:
        last_time = str(df_actual_f[datetime_col].iloc[-1])
        fig.add_vline(x=last_time, line_width=2, line_dash="dash", line_color="gray")

    fig.update_layout(
        template="plotly_white", height=450, hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=20),
        xaxis_title="Time", yaxis_title="Bike Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(showspikes=True, spikemode="across", spikethickness=1, spikecolor="#999", spikedash="dash"),
    )
    return fig