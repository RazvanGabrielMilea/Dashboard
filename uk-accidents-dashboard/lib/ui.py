import streamlit as st
from lib.state import get_filters, set_filters, reset_filters
from lib.data import get_sidebar_metadata

def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2.2rem; padding-bottom: 2.2rem; max-width: 1280px; }
        section[data-testid="stSidebar"] { padding-top: 0.8rem; }
        [data-testid="stAppViewContainer"] { background: radial-gradient(1200px 600px at 20% 0%, rgba(255,255,255,0.06), transparent 60%); }
        .side-h { font-size: 1.1rem; font-weight: 750; margin: 0.3rem 0 0.6rem; }
        .panel-title { font-size: 1.05rem; font-weight: 700; margin: 0 0 0.4rem; }
        div[data-testid="stVerticalBlockBorderWrapper"]{
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.03);
            box-shadow: 0 8px 30px rgba(0,0,0,0.18);
        }
        .stButton button { border-radius: 12px; border: 1px solid rgba(255,255,255,0.12); }
        .stMultiSelect span[data-baseweb="tag"]{ border-radius: 999px !important; }
        div[data-testid="stVerticalBlock"] > div { gap: 0.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def header():
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

def render_sidebar_filters():
    f = get_filters()

    with st.sidebar:
        st.markdown("<div class='side-h'>Filters</div>", unsafe_allow_html=True)

        meta = get_sidebar_metadata()
        years = meta.get("years", [2021])
        sev_all = meta.get("severity_labels", ["Slight", "Serious", "Fatal"])
        lsoas = meta.get("top_lsoas", ["All"])

        # Year
        default_idx = years.index(f["year"]) if f.get("year") in years else len(years) - 1
        year = st.selectbox("Year", options=years, index=default_idx)

        # LSOA
        default_lsoa = f.get("lsoa", "All")
        lsoa_idx = lsoas.index(default_lsoa) if default_lsoa in lsoas else 0
        lsoa = st.selectbox("LSOA of casualty (top 200)", options=lsoas, index=lsoa_idx)

        # Severity
        default_sev = [s for s in f.get("severity", []) if s in sev_all] or sev_all
        severity = st.multiselect("Collision severity", options=sev_all, default=default_sev)

        st.markdown("")
        if st.button("Apply", use_container_width=True):
            set_filters({"year": year, "lsoa": lsoa, "severity": severity})
            st.toast("Filters applied")
        if st.button("Reset", use_container_width=True):
            reset_filters()
            st.toast("Filters reset")
            st.rerun()
