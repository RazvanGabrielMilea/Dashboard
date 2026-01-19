"""
Main Streamlit application for UK Road Accidents Dashboard.

This is the entry point for the UK Road Accidents Dashboard application.
It configures the Streamlit app settings, applies custom styling, renders
the sidebar filters, and sets up multi-page navigation for all dashboard
sections including Overview, Trends, Hotspot Map, Factors & Severity,
Explorer, and ML Insights.
"""

import streamlit as st
from lib.ui import inject_css, render_sidebar_filters, header

# Configure the Streamlit page settings
st.set_page_config(
    page_title="UK Road Accidents Dashboard", page_icon="🚗", layout="wide"
)

# Apply custom CSS styling to the application
inject_css()
# Render the main header section
header()
# Render the sidebar with filter controls
render_sidebar_filters()

# Define the page navigation structure for the multi-page application
pages = {
    "Dashboard": [
        st.Page("pages/1_Overview.py", title="Overview", icon="📊"),
        st.Page("pages/2_Trends.py", title="Trends", icon="📈"),
        st.Page("pages/3_Hotspots_Map.py", title="Hotspot Map", icon="🗺️"),
        st.Page("pages/4_Factors_Severity.py", title="Factors & Severity", icon="⚠️"),
        st.Page("pages/5_Explorer.py", title="Explorer", icon="🧾"),
        st.Page("pages/6_ML_Insights.py", title="ML & Correlations", icon="💻"),
    ]
}

# Create and run the navigation system with expanded menu
pg = st.navigation(pages, expanded=True)
pg.run()
