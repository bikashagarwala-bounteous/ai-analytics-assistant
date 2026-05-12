"""
Analytics Dashboard — live metrics, trends, and top failure intents.
Auto-refreshes every 60 seconds.
"""

import time
from datetime import datetime

import streamlit as st

from components.api_client import get
from components.charts import (
    line_chart, bar_chart, gauge_chart, donut_chart
)


def render():
    st.title("Analytics Dashboard")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_days, col_refresh, col_auto, col_ts = st.columns([2, 1, 1, 2])
    with col_days:
        days = st.selectbox("Period", [7, 14, 30, 90], index=0, label_visibility="collapsed")
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            st.session_state["dashboard_force"] = True
            st.session_state.dashboard_last_refresh = 0
            st.rerun()
    with col_auto:
        auto_refresh = st.toggle("Auto-refresh", value=False)
    with col_ts:
        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    # ── Fetch data ────────────────────────────────────────────────────────────
    force = st.session_state.pop("dashboard_force", False)
    with st.spinner("Loading metrics..."):
        data = get("/dashboard", params={"days": days, "force": force})

    if not data:
        st.error("Could not load dashboard data. Check that the backend is running.")
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.subheader("Overview")
    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Sessions", f"{data['total_sessions']:,}")
    k2.metric("Messages", f"{data['total_messages']:,}")
    k3.metric("Engagement", f"{data['engagement_rate']:.1f}%")
    k4.metric("Failure Rate", f"{data['failure_rate']:.1f}%")
    avg_r = data.get("avg_rating")
    k5.metric("Avg Rating", f"{avg_r:.1f} / 5" if avg_r else "—")

    st.divider()

    # ── Gauge row ─────────────────────────────────────────────────────────────
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            gauge_chart(data["engagement_rate"], "Engagement Rate"),
            use_container_width=True,
        )
    with g2:
        # Invert failure rate for the gauge — lower is better
        st.plotly_chart(
            gauge_chart(100 - data["failure_rate"], "Success Rate"),
            use_container_width=True,
        )

    st.divider()

    # ── Trend charts ──────────────────────────────────────────────────────────
    st.subheader("Trends")
    t1, t2, t3 = st.columns(3)

    with t1:
        st.plotly_chart(
            line_chart(
                data["engagement_trend"], "date", "value",
                "Engagement Rate", color="primary", suffix="%",
            ),
            use_container_width=True,
        )
    with t2:
        st.plotly_chart(
            line_chart(
                data["failure_trend"], "date", "value",
                "Failure Rate", color="danger", suffix="%",
            ),
            use_container_width=True,
        )
    with t3:
        st.plotly_chart(
            line_chart(
                data["volume_trend"], "date", "value",
                "Message Volume", color="success",
            ),
            use_container_width=True,
        )

    st.divider()

    # ── Intent breakdown ──────────────────────────────────────────────────────
    st.subheader("Intent Distribution")
    i1, i2 = st.columns([3, 2])

    with i1:
        st.plotly_chart(
            bar_chart(
                data["top_intents"], "intent", "count",
                "Top Intents by Volume", horizontal=True,
            ),
            use_container_width=True,
        )
    with i2:
        intents = data.get("top_intents", [])[:5]
        st.plotly_chart(
            donut_chart(
                [i["intent"] for i in intents],
                [i["count"] for i in intents],
                "Intent Share",
            ),
            use_container_width=True,
        )

    st.divider()

    # ── Top failure intents ───────────────────────────────────────────────────
    st.subheader("Top Failure Intents")
    failures = data.get("top_failures", [])
    if failures:
        cols = st.columns(len(failures[:6]))
        for col, f in zip(cols, failures[:6]):
            col.metric(
                label=f["intent"],
                value=f"{f['rate']:.1f}%",
                delta=f"{f['failures']} failures",
                delta_color="inverse",
            )
    else:
        st.info("No failure data for this period.")

    st.divider()

    # ── Recent feedback ───────────────────────────────────────────────────────
    st.subheader("Recent Feedback")
    feedback = data.get("recent_feedback", [])
    if feedback:
        for fb in feedback[:5]:
            rating_stars = "★" * fb["rating"] + "☆" * (5 - fb["rating"])
            sentiment_color = {
                "positive": "green",
                "negative": "red",
                "neutral": "gray",
            }.get(fb["sentiment"], "gray")

            with st.container(border=True):
                r1, r2 = st.columns([1, 4])
                r1.markdown(f"**{rating_stars}**")
                r2.markdown(
                    f':{sentiment_color}[{fb["sentiment"].upper()}] · '
                    f'{fb["created_at"][:10]}'
                )
                if fb.get("comment"):
                    st.caption(fb["comment"])
    else:
        st.info("No feedback submitted in this period.")

    # ── Auto-refresh — non-blocking: schedule rerun after 60 s ───────────────
    if auto_refresh:
        if "dashboard_last_refresh" not in st.session_state:
            st.session_state.dashboard_last_refresh = time.time()
        elapsed = time.time() - st.session_state.dashboard_last_refresh
        remaining = max(0, 60 - int(elapsed))
        st.caption(f"Next auto-refresh in {remaining}s")
        if elapsed >= 60:
            st.session_state.dashboard_last_refresh = time.time()
            st.rerun()
        else:
            time.sleep(1)
            st.rerun()

render()