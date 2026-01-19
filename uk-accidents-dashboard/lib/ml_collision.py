from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

import joblib

MODEL_OUT = Path("assets/ml/collision_severity_model.joblib")
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = ["Slight", "Serious", "Fatal"]
CLASS_WEIGHTS = {"Fatal": 6.0, "Serious": 3.0, "Slight": 1.0}


def _onehot():
    # sklearn compatibility (older/newer)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_collision_ml_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # target (collision-level)
    if "_collision_sev" in out.columns:
        out["target"] = out["_collision_sev"].astype(str)
    elif "collision_severity" in out.columns:
        mp = {1: "Fatal", 2: "Serious", 3: "Slight", 1.0: "Fatal", 2.0: "Serious", 3.0: "Slight"}
        out["target"] = out["collision_severity"].map(mp).fillna(out["collision_severity"].astype(str))
    else:
        out["target"] = pd.NA

    out["target"] = out["target"].where(out["target"].isin(SEVERITY_ORDER), other=pd.NA)

    # time-derived features (if present)
    if "_date" in out.columns:
        d = pd.to_datetime(out["_date"], errors="coerce")
        out["month_num"] = d.dt.month.astype("Int64")
        out["day_num"] = d.dt.day.astype("Int64")
    else:
        out["month_num"] = pd.NA
        out["day_num"] = pd.NA

    # ensure hour numeric
    if "_hour" in out.columns:
        out["_hour"] = pd.to_numeric(out["_hour"], errors="coerce")

    return out


def build_pipeline(cat_cols: list[str], num_cols: list[str], random_state: int = 42) -> Pipeline:
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _onehot())])

    preprocess = ColumnTransformer(
        [("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=22,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
        class_weight=CLASS_WEIGHTS,
    )

    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def train_evaluate_collision(
    df: pd.DataFrame,
    feature_cols: list[str],
    sample_n: int = 120_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    d = prepare_collision_ml_df(df).dropna(subset=["target"]).copy()

    # keep only existing features
    feature_cols = [c for c in feature_cols if c in d.columns]
    if not feature_cols:
        raise ValueError("No valid feature columns found for collision model.")

    # sample for speed
    if len(d) > sample_n:
        d = d.sample(sample_n, random_state=random_state)

    X = d[feature_cols].copy()
    y = d["target"].astype(str)

    # cat vs num
    cat_cols = [c for c in feature_cols if (X[c].dtype == "object" or str(X[c].dtype).startswith("category"))]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_pipeline(cat_cols=cat_cols, num_cols=num_cols, random_state=random_state)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    cm = confusion_matrix(y_test, pred, labels=SEVERITY_ORDER)
    report = classification_report(y_test, pred, labels=SEVERITY_ORDER)

    # permutation importance (approx, fast-ish)
    imp = None
    try:
        X_imp = X_test.copy()
        if len(X_imp) > 20_000:
            X_imp = X_imp.sample(20_000, random_state=random_state)
        r = permutation_importance(model, X_imp, y_test.loc[X_imp.index], n_repeats=5, random_state=random_state, n_jobs=-1)
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
    }


def save_model(model: Pipeline, features: list[str]) -> None:
    joblib.dump({"model": model, "features": features}, MODEL_OUT)


def load_model():
    if not MODEL_OUT.exists():
        return None
    obj = joblib.load(MODEL_OUT)
    return obj["model"], obj["features"]
