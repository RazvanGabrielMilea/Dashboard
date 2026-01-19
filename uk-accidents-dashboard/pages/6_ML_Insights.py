import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from lib.data import load_data, apply_filters
from lib.state import get_filters

from lib.ml_collision import (
    train_evaluate_collision,
    save_model as save_collision_model,
    load_model as load_collision_model,
)

from lib.ml_team import (
    load_ml_cache,
    train_multiclass,
    train_binary,
    save_model as save_casualty_model,
    load_model as load_casualty_model,
)

st.set_page_config(layout="wide")

"""
ML Insights Page for UK Accidents Dashboard.

This Streamlit page provides machine learning tools for analyzing casualty and collision severity.
It includes correlation analysis, model training, evaluation, risk insights, and what-if predictions.
The page is divided into two tabs: one for casualty-level models and one for collision-level models.
"""


def _risk_insights_block(
    *,
    title: str,
    base_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    proba_label: str,
    class_selector,
    extra_show_cols: list[str] | None = None,
    key_prefix: str = "risk",
):
    """
    Generates a block for displaying top predicted risk cases and their characteristics.

    This function creates an interactive Streamlit interface that allows users to sample
    the dataset, predict probabilities using the provided model, and display the top
    high-risk rows based on the selected class probability. It also provides a breakdown
    of categorical features for the high-risk rows.

    Parameters:
    - title (str): The title to display for this insights block.
    - base_df (pd.DataFrame): The base DataFrame containing the data to analyze.
    - model: The trained machine learning model with a predict_proba method.
    - feature_cols (list[str]): List of feature column names used by the model.
    - proba_label (str): Label for the probability column in the display (e.g., "P(Fatal)").
    - class_selector (callable): A function that takes the list of classes and returns
      the target class value for which to compute probabilities.
    - extra_show_cols (list[str] | None): Additional columns to display in the table.
    - key_prefix (str): Prefix for Streamlit widget keys to avoid conflicts.

    Returns:
    None: This function displays content in the Streamlit app and does not return a value.
    """
    st.markdown(f"<div class='panel-title'>{title}</div>", unsafe_allow_html=True)
    with st.container(border=True):
        if base_df.empty:
            st.info("No rows available for insights.")
            return

        # Reset index to avoid alignment surprises
        work = base_df.reset_index(drop=True).copy()

        # Make sure features exist in df
        feats = [c for c in feature_cols if c in work.columns]
        if not feats:
            st.warning("No feature columns found in the current dataset.")
            return

        # Controls
        c1, c2, c3 = st.columns([1, 1, 2])
        sample_n = c1.slider(
            "Sample size",
            5_000,
            200_000,
            20_000,
            step=5_000,
            key=f"{key_prefix}_sample",
        )
        top_k = c2.slider(
            "Top K high-risk rows", 10, 300, 50, step=10, key=f"{key_prefix}_topk"
        )
        seed = c3.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            key=f"{key_prefix}_seed",
        )

        run = st.button(
            "Generate insights",
            type="primary",
            use_container_width=True,
            key=f"{key_prefix}_run",
        )

        if run:
            if len(work) > int(sample_n):
                work = work.sample(int(sample_n), random_state=int(seed))

            # Predict probabilities
            X = work[feats].copy()
            probas = model.predict_proba(X)
            classes = list(model.classes_)
            target_class = class_selector(classes)

            if target_class not in classes:
                st.error(f"Target class {target_class} not in model classes: {classes}")
                return

            idx = classes.index(target_class)
            work["_risk_proba"] = probas[:, idx]

            # Table columns to show
            show_cols: list[str] = []
            if extra_show_cols:
                show_cols.extend([c for c in extra_show_cols if c in work.columns])

            # Always include features (but avoid duplicates)
            for c in feats:
                if c not in show_cols:
                    show_cols.append(c)

            show_cols = ["__risk__"] + show_cols  # placeholder for display

            out = (
                work.sort_values("_risk_proba", ascending=False).head(int(top_k)).copy()
            )
            out["__risk__"] = (out["_risk_proba"] * 100.0).round(2)  # % for display

            # Keep only display cols that exist
            display_cols = [c for c in show_cols if c in out.columns]

            st.write(f"Top rows by **{proba_label}** (from a sample of {len(work):,}):")
            st.dataframe(out[display_cols], use_container_width=True, height=420)

            # Download
            csv_bytes = out[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download top-risk rows (CSV)",
                data=csv_bytes,
                file_name=f"{key_prefix}_top_risk.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("")
            st.markdown(
                "<div class='panel-title'>What characterizes the high-risk rows?</div>",
                unsafe_allow_html=True,
            )

            # Simple distribution charts for categorical columns (if present)
            cat_candidates = [
                "weather_conditions",
                "light_conditions",
                "road_surface_conditions",
                "_dow",
                "lsoa_of_casualty",
                "vehicle_type",
                "sex_of_driver",
            ]
            cats = [c for c in cat_candidates if c in out.columns]

            if not cats:
                st.caption(
                    "No categorical columns found for distributions (in the current dataset)."
                )
                return

            pick = st.selectbox(
                "Breakdown column", options=cats, key=f"{key_prefix}_breakdown"
            )
            vc = out[pick].astype(str).value_counts(dropna=False).head(15).reset_index()
            vc.columns = ["value", "count"]

            fig = px.bar(vc, x="value", y="count")
            fig.update_layout(xaxis_title=pick, yaxis_title="Count in top-risk rows")
            fig.update_traces(
                hovertemplate=f"{pick}: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)


def _safe_default_for_feature(feat: str):
    """
    Provides a safe default value for a given feature based on typical traffic accident data.

    This function returns predefined default values for various features commonly used
    in accident severity models. These defaults are chosen to be reasonable and natural
    for the traffic domain, helping to initialize forms and avoid errors.

    Parameters:
    - feat (str): The name of the feature for which to get a default value.

    Returns:
    The default value for the feature. Types vary: int for numeric features,
    float for coordinates, str for empty defaults.
    """
    # Simple defaults that feel natural in traffic domain
    defaults = {
        "_hour": 17,
        "month_num": 6,
        "day_num": 1,
        "speed_limit": 30,
        "age_of_driver": 35,
        "age_of_vehicle": 8,
        "engine_capacity_cc": 1600,
        "age_of_casualty": 30,
        "latitude": 51.5074,  # London-ish
        "longitude": -0.1278,
        "persons_involved": 2,
        "persons_killed": 0,
    }
    return defaults.get(feat, "")


# Preset scenarios for what-if predictions, providing common traffic accident contexts
SCENARIO_PRESETS = {
    "Custom (manual)": {},
    "Rush hour (weekday)": {"_dow": "Friday", "_hour": 17, "speed_limit": 30},
    "Night + rain + wet road": {
        "_dow": "Saturday",
        "_hour": 23,
        "weather_conditions": "2",  # if your dataset stores codes as strings
        "light_conditions": "4",
        "road_surface_conditions": "2",
        "speed_limit": 30,
    },
    "High-speed road (daylight)": {
        "_dow": "Monday",
        "_hour": 12,
        "speed_limit": 70,
        "light_conditions": "1",
        "road_surface_conditions": "1",
    },
    "Early morning commute": {"_dow": "Tuesday", "_hour": 8, "speed_limit": 30},
}


def _what_if_block(
    *,
    title: str,
    model,
    feature_cols: list[str],
    base_df: pd.DataFrame | None,
    key_prefix: str,
):
    """
    Renders a 'What-if' prediction form for a trained machine learning model.

    This function creates an interactive Streamlit form that allows users to input
    values for model features and get predictions. It includes preset scenarios
    for quick setup and dynamically generates input widgets based on feature types.

    Parameters:
    - title (str): The title to display for this what-if block.
    - model: The trained machine learning model with predict_proba method.
    - feature_cols (list[str]): List of feature column names used by the model.
    - base_df (pd.DataFrame | None): Optional DataFrame to derive categorical options
      from existing data.
    - key_prefix (str): Prefix for Streamlit widget keys to avoid conflicts.

    Returns:
    None: This function displays content in the Streamlit app and does not return a value.
    """
    st.markdown(f"<div class='panel-title'>{title}</div>", unsafe_allow_html=True)
    with st.container(border=True):
        if model is None or not feature_cols:
            st.info("Train / load a model first to enable What-if predictions.")
            return

        # options for categorical features from base_df if provided
        def cat_options(col: str):
            if base_df is None or col not in base_df.columns:
                return []
            vals = base_df[col].dropna().astype(str).unique().tolist()
            vals = sorted(vals)[:200]  # cap for UI
            return vals

        # feature typing heuristic
        numeric_like = {
            "_hour",
            "month_num",
            "day_num",
            "speed_limit",
            "age_of_driver",
            "age_of_vehicle",
            "engine_capacity_cc",
            "age_of_casualty",
            "latitude",
            "longitude",
            "persons_involved",
            "persons_killed",
        }

        num_feats = [c for c in feature_cols if c in numeric_like]
        cat_feats = [c for c in feature_cols if c not in numeric_like]

        # ---------------------------
        # Scenario preset selector
        # ---------------------------
        preset_name = st.selectbox(
            "Scenario preset",
            list(SCENARIO_PRESETS.keys()),
            index=0,
            key=f"{key_prefix}_preset",
            help="Choose a preset to auto-fill values, then tweak anything you want and click Predict.",
        )
        preset = SCENARIO_PRESETS.get(preset_name, {})

        # Build row with defaults first
        row = {c: _safe_default_for_feature(c) for c in feature_cols}
        # Override with preset values (only for columns used by the model)
        for k, v in preset.items():
            if k in row:
                row[k] = v

        st.caption(
            "Fill inputs → click Predict. Only features used by the trained model are shown."
        )

        # ---------------------------
        # Form
        # ---------------------------
        with st.form(f"{key_prefix}_whatif_form"):
            # Numeric inputs
            if num_feats:
                st.markdown("**Numeric inputs**")
                cols = st.columns(4)
                for i, feat in enumerate(num_feats):
                    val = row.get(feat, _safe_default_for_feature(feat))

                    if feat == "_hour":
                        row[feat] = cols[i % 4].number_input("Hour", 0, 23, int(val), 1)
                    elif feat == "month_num":
                        row[feat] = cols[i % 4].number_input(
                            "Month", 1, 12, int(val), 1
                        )
                    elif feat == "day_num":
                        row[feat] = cols[i % 4].number_input(
                            "Day of month", 1, 31, int(val), 1
                        )
                    elif feat == "persons_involved":
                        row[feat] = cols[i % 4].number_input(
                            "Persons involved", 1, 100, int(val), 1
                        )
                    elif feat == "persons_killed":
                        row[feat] = cols[i % 4].number_input(
                            "Persons killed", 0, 50, int(val), 1
                        )
                    elif feat == "speed_limit":
                        row[feat] = cols[i % 4].number_input(
                            "Speed limit", 0, 140, int(val), 5
                        )
                    elif feat in ["age_of_driver", "age_of_vehicle", "age_of_casualty"]:
                        row[feat] = cols[i % 4].number_input(
                            feat.replace("_", " ").title(), 0, 120, int(val), 1
                        )
                    elif feat == "engine_capacity_cc":
                        row[feat] = cols[i % 4].number_input(
                            "Engine capacity (cc)", 0, 10000, int(val), 100
                        )
                    elif feat in ["latitude", "longitude"]:
                        row[feat] = cols[i % 4].number_input(
                            feat.title(), value=float(val)
                        )
                    else:
                        row[feat] = cols[i % 4].number_input(
                            feat, value=float(val) if val != "" else 0.0
                        )

                st.markdown("")

            # Categorical inputs
            if cat_feats:
                st.markdown("**Categorical inputs**")
                cols = st.columns(3)
                for i, feat in enumerate(cat_feats):
                    label = feat.replace("_", " ").title()
                    opts = cat_options(feat)

                    if feat == "_dow":
                        dow = [
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                            "Saturday",
                            "Sunday",
                        ]
                        # preset value if present
                        preset_dow = str(row.get(feat, "Friday"))
                        idx = dow.index(preset_dow) if preset_dow in dow else 0
                        row[feat] = cols[i % 3].selectbox("Day of week", dow, index=idx)
                    elif opts:
                        preset_val = str(row.get(feat, opts[0]))
                        idx = opts.index(preset_val) if preset_val in opts else 0
                        row[feat] = cols[i % 3].selectbox(label, opts, index=idx)
                    else:
                        row[feat] = cols[i % 3].text_input(
                            label, value=str(row.get(feat, ""))
                        )

                st.markdown("")

            go = st.form_submit_button(
                "Predict", type="primary", use_container_width=True
            )

        if go:
            X = pd.DataFrame([{c: row.get(c, None) for c in feature_cols}])

            try:
                proba = model.predict_proba(X)[0]
                classes = list(model.classes_)
                out = pd.DataFrame(
                    {"class": classes, "probability": proba}
                ).sort_values("probability", ascending=False)
                out["probability"] = (out["probability"] * 100.0).round(2)

                st.write("Predicted probabilities (%):")
                st.dataframe(out, use_container_width=True, height=240)
                st.success(f"Prediction: **{out.iloc[0]['class']}**")
            except Exception as e:
                st.error(
                    "Prediction failed. This usually happens if the scenario uses a categorical value "
                    "never seen during training. Try another preset or choose values from dropdowns."
                )
                st.exception(e)


# Main page header
st.markdown("<div class='panel-title'>ML & Correlations</div>", unsafe_allow_html=True)

# Create tabs for casualty-level and collision-level models
tabs = st.tabs(["Casualty severity (row-level)", "Collision severity (accident-level)"])

# ============================================================
# TAB 1: CASUALTY SEVERITY (Team model)
# ============================================================
with tabs[0]:
    # Load the preprocessed ML cache data for casualty severity models
    df = load_ml_cache()

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows (ML cache)", f"{len(df):,}")
        c2.metric("Target", "casualty_severity")
        c3.metric("Saved model", "Yes" if load_casualty_model() else "No")

    st.markdown("")

    # Section: Analyze correlations between numeric features and casualty severity
    st.markdown(
        "<div class='panel-title'>Correlations (Spearman)</div>", unsafe_allow_html=True
    )
    with st.container(border=True):
        num_candidates = [
            "latitude",
            "longitude",
            "speed_limit",
            "age_of_vehicle",
            "age_of_driver",
            "engine_capacity_cc",
            "age_of_casualty",
        ]
        cols = [c for c in num_candidates if c in df.columns]

        corr_n = st.slider(
            "Correlation sample size",
            10_000,
            300_000,
            80_000,
            step=10_000,
            key="corr_n_casualty",
        )

        d = df[cols + ["casualty_severity"]].dropna().copy()
        if len(d) > corr_n:
            d = d.sample(corr_n, random_state=42)

        d["severity_num"] = pd.to_numeric(d["casualty_severity"], errors="coerce")
        corr_cols = cols + ["severity_num"]
        corr = d[corr_cols].corr(method="spearman", numeric_only=True)

        fig = px.imshow(corr, text_auto=".2f", aspect="auto")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        if "severity_num" in corr.columns:
            s = (
                corr["severity_num"]
                .drop("severity_num")
                .sort_values(key=lambda x: np.abs(x), ascending=False)
            )
            st.caption("Strongest correlations with severity:")
            st.write(s.head(5))

    st.markdown("")

    # Section: Train and evaluate casualty severity models
    st.markdown("<div class='panel-title'>Train models</div>", unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])

        with col1:
            sample_n = st.slider(
                "Training sample size",
                20_000,
                300_000,
                150_000,
                step=10_000,
                key="sample_n_casualty",
            )

            train_mc = st.button(
                "Train multiclass (Fatal/Serious/Slight)",
                type="primary",
                use_container_width=True,
                key="train_mc_btn",
            )
            train_bin = st.button(
                "Train binary (Serious/Fatal vs Slight)",
                use_container_width=True,
                key="train_bin_btn",
            )

        with col2:
            st.info(
                "Training runs only when you click.\n"
                "Multiclass matches your notebook weights.\n"
                "Binary matches your Model3 notebook."
            )

    if "cas_ml_res" not in st.session_state:
        st.session_state["cas_ml_res"] = None
    if "cas_ml_kind" not in st.session_state:
        st.session_state["cas_ml_kind"] = None

    if train_mc:
        with st.spinner("Training multiclass model..."):
            res = train_multiclass(df, sample_n=int(sample_n))
        st.session_state["cas_ml_res"] = res
        st.session_state["cas_ml_kind"] = "multiclass"
        save_casualty_model(res["model"], res["features"])
        st.success("Multiclass model trained + saved ✅")

    if train_bin:
        with st.spinner("Training binary model..."):
            res = train_binary(df, sample_n=int(sample_n))
        st.session_state["cas_ml_res"] = res
        st.session_state["cas_ml_kind"] = "binary"
        save_casualty_model(res["model"], res["features"])
        st.success("Binary model trained + saved ✅")

    res = st.session_state.get("cas_ml_res")
    kind = st.session_state.get("cas_ml_kind")

    # Display model results if a model has been trained
    if res:
        st.markdown("<div class='panel-title'>Results</div>", unsafe_allow_html=True)
        with st.container(border=True):
            if kind == "multiclass":
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{res['acc']*100:.2f}%")
                c2.metric("F1 (macro)", f"{res['f1_macro']:.3f}")
                c3.metric("Train/Test", f"{res['n_train']:,} / {res['n_test']:,}")

                cm_df = pd.DataFrame(
                    res["cm"],
                    index=["Fatal", "Serious", "Slight"],
                    columns=["Fatal", "Serious", "Slight"],
                )
                fig = px.imshow(cm_df, text_auto=True, aspect="auto")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Classification report"):
                    st.text(res["report"])

                imp = res.get("importance")
                if imp is not None and not imp.empty:
                    st.markdown(
                        "<div class='panel-title'>Permutation importance</div>",
                        unsafe_allow_html=True,
                    )
                    top = imp.head(12)
                    fig = px.bar(top, x="feature", y="importance")
                    fig.update_layout(
                        xaxis_title="Feature", yaxis_title="Importance (mean)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{res['acc']*100:.2f}%")
                c2.metric("F1", f"{res['f1']:.3f}")
                c3.metric("ROC AUC", f"{res['auc']:.3f}")

                cm_df = pd.DataFrame(
                    res["cm"],
                    index=["Not serious", "Serious/Fatal"],
                    columns=["Pred 0", "Pred 1"],
                )
                fig = px.imshow(cm_df, text_auto=True, aspect="auto")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Classification report"):
                    st.text(res["report"])

    # Section: Generate risk insights using the trained model
    # ----------------------------
    # Model-driven insights (Casualty)
    # ----------------------------
    cas_res = st.session_state.get("cas_ml_res")
    cas_kind = st.session_state.get("cas_ml_kind")

    if cas_res is None:
        st.info("Train a casualty model to generate high-risk insights.")
    else:
        model = cas_res["model"]
        feats = cas_res["features"]

        extra_cols = [
            "casualty_severity",
            "speed_limit",
            "weather_conditions",
            "light_conditions",
            "road_surface_conditions",
            "vehicle_type",
            "sex_of_driver",
        ]

        if cas_kind == "multiclass":
            _risk_insights_block(
                title="High-risk cases (Predicted Fatal – casualty model)",
                base_df=df,
                model=model,
                feature_cols=feats,
                proba_label="P(Fatal)",
                class_selector=lambda classes: 1,
                extra_show_cols=extra_cols,
                key_prefix="cas_mult",
            )
        elif cas_kind == "binary":
            _risk_insights_block(
                title="High-risk cases (Predicted Serious/Fatal – casualty model)",
                base_df=df,
                model=model,
                feature_cols=feats,
                proba_label="P(Serious/Fatal)",
                class_selector=lambda classes: 1,
                extra_show_cols=extra_cols,
                key_prefix="cas_bin",
            )
        else:
            st.info("Train a model (multiclass or binary) to generate insights.")
    # ----------------------------
    # What-if prediction (Casualty)
    # ----------------------------
    # Section: Interactive what-if predictions for casualty models
    cas_res = st.session_state.get("cas_ml_res")
    if cas_res is None:
        st.info("Train a casualty model to enable What-if prediction.")
    else:
        _what_if_block(
            title="What-if predictor (Casualty model)",
            model=cas_res["model"],
            feature_cols=cas_res["features"],
            base_df=df,  # ml_cache for dropdown values
            key_prefix="casualty",
        )

# ============================================================
# TAB 2: COLLISION SEVERITY (Dashboard-aligned)
# ============================================================
with tabs[1]:
    # Section: Overview of collision severity model aligned with dashboard
    st.markdown(
        "<div class='panel-title'>Collision severity model (matches dashboard)</div>",
        unsafe_allow_html=True,
    )

    # Load and filter collision data based on current dashboard filters
    filters = get_filters()
    df_col = load_data()
    df_col_f = apply_filters(df_col, filters)

    # add derived fields if missing (UI convenience; trainer also handles this)
    if "_date" in df_col_f.columns:
        dd = pd.to_datetime(df_col_f["_date"], errors="coerce")
        if "month_num" not in df_col_f.columns:
            df_col_f = df_col_f.copy()
            df_col_f["month_num"] = dd.dt.month.astype("Int64")
            df_col_f["day_num"] = dd.dt.day.astype("Int64")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows (filtered collisions)", f"{len(df_col_f):,}")
        c2.metric(
            "Year", "All" if filters.get("year") is None else str(filters.get("year"))
        )
        c3.metric("Saved model", "Yes" if load_collision_model() else "No")

    st.markdown("")

    candidates = [
        "_hour",
        "_dow",
        "month_num",
        "day_num",
        "persons_involved",
        "persons_killed",
        "lsoa_of_casualty",
        "weather_conditions",
        "light_conditions",
        "road_surface_conditions",
        "speed_limit",
    ]
    available = [c for c in candidates if c in df_col_f.columns]
    default_feats = [
        c for c in ["_hour", "_dow", "month_num", "persons_involved"] if c in available
    ]

    # Section: Configure and train collision severity models
    st.markdown(
        "<div class='panel-title'>Train / Evaluate</div>", unsafe_allow_html=True
    )
    with st.container(border=True):
        feature_cols = st.multiselect(
            "Feature columns",
            options=available,
            default=default_feats,
            key="collision_features",
        )

        sample_n = st.slider(
            "Training sample size",
            20_000,
            300_000,
            120_000,
            step=10_000,
            key="collision_sample_n",
        )

        test_size = st.slider(
            "Test size",
            0.1,
            0.4,
            0.2,
            step=0.05,
            key="collision_test_size",
        )

        colA, colB = st.columns(2)
        train_btn = colA.button(
            "Train collision model",
            type="primary",
            use_container_width=True,
            key="train_collision_btn",
        )
        load_btn = colB.button(
            "Load saved collision model",
            use_container_width=True,
            key="load_collision_btn",
        )

    if "collision_res" not in st.session_state:
        st.session_state["collision_res"] = None

    if load_btn:
        loaded = load_collision_model()
        if loaded is None:
            st.warning("No saved collision model found yet.")
        else:
            st.success(
                "Saved collision model exists ✅ (Use Train to evaluate a new run on current data/filters)"
            )

    if train_btn:
        if not feature_cols:
            st.error("Pick at least one feature.")
        elif df_col_f.empty:
            st.error("No rows available under the current filters.")
        else:
            with st.spinner("Training collision severity model..."):
                res2 = train_evaluate_collision(
                    df_col_f,
                    feature_cols=feature_cols,
                    sample_n=int(sample_n),
                    test_size=float(test_size),
                )
            st.session_state["collision_res"] = res2
            save_collision_model(res2["model"], res2["features"])
            st.success("Trained and saved collision model ✅")

    # Display collision model results if trained
    res2 = st.session_state.get("collision_res")
    if res2:
        st.markdown("<div class='panel-title'>Results</div>", unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{res2['acc']*100:.2f}%")
            c2.metric("F1 (macro)", f"{res2['f1_macro']:.3f}")
            c3.metric("Train/Test", f"{res2['n_train']:,} / {res2['n_test']:,}")

            cm_df = pd.DataFrame(
                res2["cm"],
                index=["Slight", "Serious", "Fatal"],
                columns=["Slight", "Serious", "Fatal"],
            )
            fig = px.imshow(cm_df, text_auto=True, aspect="auto")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Classification report"):
                st.text(res2["report"])

            imp = res2.get("importance")
            if imp is not None and not imp.empty:
                st.markdown(
                    "<div class='panel-title'>Permutation importance</div>",
                    unsafe_allow_html=True,
                )
                top = imp.head(12)
                fig = px.bar(top, x="feature", y="importance")
                fig.update_layout(
                    xaxis_title="Feature", yaxis_title="Importance (mean)"
                )
                st.plotly_chart(fig, use_container_width=True)

    # Section: Generate risk insights for collision models
    # ----------------------------
    # Model-driven insights (Collision)
    # ----------------------------
    col_res = st.session_state.get("collision_res")

    if col_res is None:
        st.info("Train a collision model to generate high-risk insights.")
    else:
        model = col_res["model"]
        feats = col_res["features"]

        extra_cols = [
            "collision_index",
            "_collision_sev",
            "collision_severity",
            "_dow",
            "_hour",
            "_year",
            "_month",
            "persons_involved",
            "persons_killed",
            "weather_conditions",
            "light_conditions",
            "road_surface_conditions",
            "speed_limit",
            "lsoa_of_casualty",
        ]

        _risk_insights_block(
            title="High-risk accidents (Predicted Fatal – collision model)",
            base_df=df_col_f,
            model=model,
            feature_cols=feats,
            proba_label="P(Fatal)",
            class_selector=lambda classes: "Fatal",
            extra_show_cols=extra_cols,
            key_prefix="col_fatal",
        )

    # ----------------------------
    # What-if prediction (Collision)
    # ----------------------------
    # Section: Interactive what-if predictions for collision models
    col_res = st.session_state.get("collision_res")
    if col_res is None:
        st.info("Train a collision model to enable What-if prediction.")
    else:
        _what_if_block(
            title="What-if predictor (Collision model)",
            model=col_res["model"],
            feature_cols=col_res["features"],
            base_df=df_col_f,  # filtered collision data for dropdown values
            key_prefix="collision",
        )
