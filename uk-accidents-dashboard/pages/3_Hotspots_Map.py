import streamlit as st
from lib.assets import asset_exists, render_html

"""
Hotspot Map Page for UK Accidents Dashboard.

This Streamlit page displays interactive maps showing accident hotspots and clusters.
It provides multiple map types (grid, heatmap, cluster, cells) loaded from pre-generated
HTML files created by team members. Maps are loaded on demand to avoid performance issues.
"""

st.set_page_config(layout="wide")

# Page header and description
st.markdown("<div class='dash-title'>Hotspot Map</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='dash-sub'>Folium maps from the team export (loaded on demand)</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Define available map options with metadata (label, filename, height, scrolling)
options = [
    ("Hotspot grid (fast)", "accidents_hotspot_grid.html", 700, False),
    ("Heatmap (medium)", "accidents_heatmap.html", 700, True),
    ("Cluster map (very large)", "accidents_cluster_map.html", 700, True),
    ("Hotspot cells (table/map)", "hotspot_cells_top.html", 650, True),
]

# Extract labels for selectbox
labels = [o[0] for o in options]
choice = st.selectbox("Choose map", labels, index=0)

# Get selected option details
label, file_name, height, scrolling = next(o for o in options if o[0] == choice)

# Check if the selected map file exists
if not asset_exists("maps", file_name):
    st.error(f"Missing map asset: assets/maps/{file_name}")
    st.stop()

# Handle large maps with warning and confirmation
if "very large" in label.lower():
    st.warning("This map is huge (~56MB). It can freeze the browser inside Streamlit.")
    if st.button("Load cluster map anyway", type="primary"):
        render_html("maps", file_name, height=height, scrolling=scrolling)
else:
    render_html("maps", file_name, height=height, scrolling=scrolling)
