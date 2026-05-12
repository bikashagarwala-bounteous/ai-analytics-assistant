"""
Reusable Plotly chart builders used across dashboard pages.
All functions return a Plotly figure ready for st.plotly_chart().
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


_COLORS = {
    "primary":   "#4F8EF7",
    "success":   "#22C55E",
    "warning":   "#F59E0B",
    "danger":    "#EF4444",
    "neutral":   "#94A3B8",
    "bg":        "#0F172A",
    "surface":   "#1E293B",
    "text":      "#F1F5F9",
}

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=_COLORS["text"], size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=False,
)


def line_chart(
    data: list[dict],
    x_key: str,
    y_key: str,
    title: str,
    color: str = "primary",
    suffix: str = "",
) -> go.Figure:
    if not data:
        return _empty_chart(title)

    df = pd.DataFrame(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_key],
        y=df[y_key],
        mode="lines+markers",
        line=dict(color=_COLORS.get(color, color), width=2),
        marker=dict(size=4),
        hovertemplate=f"%{{y:.1f}}{suffix}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.1)",
            ticksuffix=suffix,
            tickfont=dict(size=10),
        ),
        **_LAYOUT_BASE,
    )
    return fig


def bar_chart(
    data: list[dict],
    x_key: str,
    y_key: str,
    title: str,
    color: str = "primary",
    horizontal: bool = False,
) -> go.Figure:
    if not data:
        return _empty_chart(title)

    df = pd.DataFrame(data)
    if horizontal:
        fig = go.Figure(go.Bar(
            x=df[y_key],
            y=df[x_key],
            orientation="h",
            marker_color=_COLORS.get(color, color),
            hovertemplate="%{x}<extra></extra>",
        ))
        fig.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=10)))
    else:
        fig = go.Figure(go.Bar(
            x=df[x_key],
            y=df[y_key],
            marker_color=_COLORS.get(color, color),
            hovertemplate="%{y}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)"),
        **_LAYOUT_BASE,
    )
    return fig


def gauge_chart(value: float, title: str, suffix: str = "%") -> go.Figure:
    color = _COLORS["success"] if value >= 70 else _COLORS["warning"] if value >= 40 else _COLORS["danger"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix=suffix, font=dict(size=24, color=_COLORS["text"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=_COLORS["text"]),
            bar=dict(color=color),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color="rgba(239,68,68,0.15)"),
                dict(range=[40, 70], color="rgba(245,158,11,0.15)"),
                dict(range=[70, 100], color="rgba(34,197,94,0.15)"),
            ],
        ),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=200,
        **_LAYOUT_BASE,
    )
    return fig


def donut_chart(labels: list[str], values: list[float], title: str) -> go.Figure:
    if not labels:
        return _empty_chart(title)
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=[
            _COLORS["primary"], _COLORS["success"], _COLORS["warning"],
            _COLORS["danger"], _COLORS["neutral"],
        ]),
        textinfo="percent",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        **{
            **_LAYOUT_BASE,
            "title": dict(text=title, font=dict(size=13)),
            "showlegend": True,
            "legend": dict(font=dict(size=10), orientation="v"),
        }
    )
    return fig


def _empty_chart(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color=_COLORS["neutral"], size=14),
    )
    fig.update_layout(title=dict(text=title, font=dict(size=13)), **_LAYOUT_BASE)
    return fig