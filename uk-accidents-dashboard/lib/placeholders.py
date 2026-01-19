import streamlit as st

def kpi_placeholder(label: str, value: str = "—"):
    st.metric(label=label, value=value)

def panel_placeholder(title: str, height_px: int = 320, note: str = "Placeholder – chart will be inserted here."):
    st.markdown(f"<div class='panel-title'>{title}</div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.info(note)
        st.markdown(f"<div style='height:{height_px}px'></div>", unsafe_allow_html=True)
