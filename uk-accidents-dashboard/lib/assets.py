from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


DEFAULT_ASSETS_DIR = Path("assets")
ENV_ASSETS = os.getenv("UK_ACCIDENTS_ASSETS")


def assets_root() -> Path:
    return Path(ENV_ASSETS) if ENV_ASSETS else DEFAULT_ASSETS_DIR


def asset_path(*parts: str) -> Path:
    return assets_root().joinpath(*parts)


def asset_exists(*parts: str) -> bool:
    return asset_path(*parts).exists()


@st.cache_data(show_spinner=False)
def read_text(*parts: str) -> str:
    p = asset_path(*parts)
    if not p.exists():
        raise FileNotFoundError(f"Missing asset: {p.resolve()}")
    return p.read_text(encoding="utf-8", errors="ignore")


@st.cache_data(show_spinner=False)
def read_csv(*parts: str) -> pd.DataFrame:
    p = asset_path(*parts)
    if not p.exists():
        raise FileNotFoundError(f"Missing asset: {p.resolve()}")
    return pd.read_csv(p)


def render_html(*parts: str, height: int = 650, scrolling: bool = False):
    html = read_text(*parts)
    components.html(html, height=height, scrolling=scrolling)
