from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
import joblib


ML_CACHE = Path("data/ml_cache.parquet")
MODEL_OUT = Path("assets/ml/team_model.joblib")
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)


# feature sets from your notebooks
NUMERIC = [
    "latitude", "longitude",
    "speed_limit",
    "age_of_vehicle", "age_of_driver", "engine_capacity_cc",
    "age_of_casualty",
]

CATEGORICAL = [
    "weather_conditions",
    "road_surface_conditions",
    "light_conditions",
    "vehicle_type",
    "sex_of_driver",
    "propulsion_code",
    "journey_purpose_of_driver",
    "casualty_class",
    "sex_of_casualty",
]


@st.cache_data(show_spinner="Loading ML cache…")
def load_ml_cache() -> pd.DataFrame:
    if not ML_CACHE.exists():
        raise FileNotFoundError(
            f"ML cache not found at {ML_CACHE.resolve()}.\n"
            f"Run: python scripts/build_ml_cache.py"
        )
    df = pd.read_parquet(ML_CACHE)
    df.columns = [c.strip() for c in df.columns]
    return df


def make_preprocess(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("num", num_tf, numeric_features),
         ("cat", cat_tf, categorical_features)],
        remainder="drop",
    )


def train_multiclass(df: pd.DataFrame, sample_n: int = 150_000) -> dict:
    # target: 1 Fatal, 2 Serious, 3 Slight (STATS19)
    d = df.dropna(subset=["casualty_severity"]).copy()
    if len(d) > sample_n:
        d = d.sample(sample_n, random_state=42)

    feature_cols = [c for c in (NUMERIC + CATEGORICAL) if c in d.columns]
    num = [c for c in NUMERIC if c in feature_cols]
    cat = [c for c in CATEGORICAL if c in feature_cols]

    X = d[feature_cols]
    y = pd.to_numeric(d["casualty_severity"], errors="coerce").astype("Int64")
    d = d[y.notna()].copy()
    X = d[feature_cols]
    y = pd.to_numeric(d["casualty_severity"], errors="coerce").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = make_preprocess(num, cat)

    # class weights from your notebook (more focus on Fatal/Serious)
    class_weights = {1: 6.0, 2: 3.0, 3: 1.0}

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=22,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weights,
    )

    model = Pipeline([("preprocess", preprocess), ("rf", rf)])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    labels = [1, 2, 3]
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    cm = confusion_matrix(y_test, pred, labels=labels)
    report = classification_report(y_test, pred, labels=labels)

    # permutation importance (fast-ish)
    imp = None
    try:
        X_imp = X_test
        if len(X_imp) > 20000:
            X_imp = X_imp.sample(20000, random_state=42)
        r = permutation_importance(model, X_imp, y_test.loc[X_imp.index], n_repeats=5, random_state=42, n_jobs=-1)
        imp = pd.DataFrame({"feature": feature_cols, "importance": r.importances_mean}).sort_values("importance", ascending=False)
    except Exception:
        pass

    return {
        "model": model,
        "features": feature_cols,
        "acc": acc,
        "f1_macro": f1m,
        "cm": cm,
        "report": report,
        "importance": imp,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "labels": labels,
    }


def train_binary(df: pd.DataFrame, sample_n: int = 150_000) -> dict:
    # binary target: serious/fatal vs slight
    d = df.dropna(subset=["casualty_severity"]).copy()
    if len(d) > sample_n:
        d = d.sample(sample_n, random_state=42)

    sev = pd.to_numeric(d["casualty_severity"], errors="coerce")
    d = d[sev.notna()].copy()
    sev = pd.to_numeric(d["casualty_severity"], errors="coerce").astype(int)
    d["is_serious"] = sev.isin([1, 2]).astype(int)

    feature_cols = [c for c in (NUMERIC + CATEGORICAL) if c in d.columns]
    num = [c for c in NUMERIC if c in feature_cols]
    cat = [c for c in CATEGORICAL if c in feature_cols]

    X = d[feature_cols]
    y = d["is_serious"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = make_preprocess(num, cat)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model = Pipeline([("preprocess", preprocess), ("rf", rf)])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="binary")
    auc = roc_auc_score(y_test, proba)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    report = classification_report(y_test, pred, labels=[0, 1])

    return {
        "model": model,
        "features": feature_cols,
        "acc": acc,
        "f1": f1m,
        "auc": auc,
        "cm": cm,
        "report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "labels": [0, 1],
    }


def save_model(model: Pipeline, features: list[str]) -> None:
    joblib.dump({"model": model, "features": features}, MODEL_OUT)


def load_model():
    if not MODEL_OUT.exists():
        return None
    obj = joblib.load(MODEL_OUT)
    return obj["model"], obj["features"]
