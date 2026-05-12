"""
Streamlit entry point.

Uses st.navigation() to handle routing between pages.
"""

import streamlit as st

st.set_page_config(
    page_title="AI Analytics Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { min-width: 220px; max-width: 220px; }
    .stMetric label { font-size: 0.75rem; color: #94A3B8; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .stButton button { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

dashboard_page = st.Page("pages/dashboard.py", title="Dashboard", icon=":material/bar_chart:")
chat_page      = st.Page("pages/chat.py",      title="Chat",      icon=":material/chat:")
lab_page       = st.Page("pages/prompt_lab.py", title="Prompt Lab", icon=":material/science:")

pg = st.navigation([dashboard_page, chat_page, lab_page])
pg.run()