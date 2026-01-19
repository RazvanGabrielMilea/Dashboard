import pandas as pd
import streamlit as st
from lib.assets import read_csv

@st.cache_data(show_spinner=False)
def load_monthly_by_year() -> pd.DataFrame:
    return read_csv("data", "overview", "monthly_by_year.csv")

@st.cache_data(show_spinner=False)
def load_hour_dow_by_year() -> pd.DataFrame:
    return read_csv("data", "overview", "hour_dow_by_year.csv")

@st.cache_data(show_spinner=False)
def load_severity_by_year() -> pd.DataFrame:
    return read_csv("data", "overview", "severity_by_year.csv")
