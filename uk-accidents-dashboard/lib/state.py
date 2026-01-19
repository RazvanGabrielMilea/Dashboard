import streamlit as st

DEFAULTS = {
    "year": 2021,
    "county": "All",
    "severity": ["Slight", "Serious", "Fatal"],
}

def init_state():
    if "filters" not in st.session_state:
        st.session_state["filters"] = DEFAULTS.copy()

def get_filters() -> dict:
    init_state()
    return st.session_state["filters"]

def set_filters(new_filters: dict):
    st.session_state["filters"] = new_filters

def reset_filters():
    st.session_state["filters"] = DEFAULTS.copy()
