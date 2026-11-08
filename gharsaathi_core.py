"""GharSaathi core logic.

Data loading, model training, price formatting, budget parsing, the KNN home
finder, EMI maths and the SQLite payment ledger. Nothing in this file imports
Streamlit, so every function can be tested on its own.
"""
from __future__ import annotations

import io
import re
import sqlite3
import urllib.request
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# The same 7 features, split and seed that the original notebook uses.
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
]
TARGET = "SalePrice"

# Public copy of Kaggle's "House Prices" train.csv (1460 rows, 81 columns).
DATA_URLS = ["http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _has_columns(df: pd.DataFrame | None) -> bool:
    return df is not None and set(FEATURES + [TARGET]).issubset(df.columns)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    keep = FEATURES + [TARGET] + (["Id"] if "Id" in df.columns else [])
    out = df[keep].copy()
    out[FEATURES + [TARGET]] = out[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    return out.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def load_data(base_dir=None):
    """Return (DataFrame, source_text). DataFrame is None if nothing worked."""
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parent

    local = base / "train.csv"
    if local.exists():
        try:
            df = pd.read_csv(local)
            if _has_columns(df):
                return _clean(df), "train.csv (file in the repository)"
        except Exception:
            pass

    for url in DATA_URLS:
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                df = pd.read_csv(io.BytesIO(resp.read()))
            if _has_columns(df):
                return _clean(df), "downloaded from a public copy of Kaggle's train.csv"
        except Exception:
            pass

    try:
        from sklearn.datasets import fetch_openml

        df = fetch_openml(name="house_prices", as_frame=True).frame
        if _has_columns(df):
            return _clean(df), "OpenML dataset 'house_prices'"
    except Exception:
        pass

    return None, "unavailable"


def dataframe_from_upload(file) -> pd.DataFrame | None:
    """Read an uploaded train.csv. Returns None if the columns are missing."""
    try:
        df = pd.read_csv(file)
    except Exception:
        return None
    return _clean(df) if _has_columns(df) else None


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def train_models(df: pd.DataFrame) -> dict:
    """Linear Regression vs Random Forest, 80/20 split, random_state=42."""
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(
        X_train, y_train
    )

    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)

    metrics = {
        "Linear Regression": {
            "mae": float(mean_absolute_error(y_test, pred_lr)),
            "r2": float(r2_score(y_test, pred_lr)),
        },
        "Random Forest": {
            "mae": float(mean_absolute_error(y_test, pred_rf)),
            "r2": float(r2_score(y_test, pred_rf)),
        },
    }
    comparison = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Linear Regression": pred_lr,
            "Random Forest": pred_rf,
        }
    )
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(
        ascending=False
    )
    return {
        "lr": lr,
        "rf": rf,
        "metrics": metrics,
        "comparison": comparison,
        "importances": importances,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def predict_price(models: dict, values: dict) -> dict:
    """Predicted price in USD from both models (never below zero)."""
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    return {
        "Linear Regression": max(float(models["lr"].predict(row)[0]), 0.0),
        "Random Forest": max(float(models["rf"].predict(row)[0]), 0.0),
    }


# --------------------------------------------------------------------------
# Money helpers
# --------------------------------------------------------------------------
def fmt_inr(amount: float) -> str:
    a = float(amount)
    if abs(a) >= 1e7:
        return f"₹{a / 1e7:.2f} Cr"
    if abs(a) >= 1e5:
        return f"₹{a / 1e5:.2f} Lakh"
    return f"₹{a:,.0f}"


_CRORE = {"cr", "crore", "crores", "करोड़", "करोड", "करोड़ों"}
_LAKH = {"l", "lac", "lacs", "lakh", "lakhs", "लाख"}
_THOUSAND = {"k", "thousand", "हज़ार", "हजार"}


def parse_budget(text: str) -> float | None:
    """'80 lakh', '1.2 crore', '15000000', '50' -> rupees. None if unreadable.

    A bare number below 1000 (for example '50') is read as lakh.
    """
    if not text:
        return None
    t = text.lower().replace(",", "")
    t = re.sub(r"₹|rs\.?|inr", " ", t)
    m = re.search(r"(\d+(?:\.\d+)?)\s*([a-zऀ-ॿ]*)", t)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2)
    if unit in _CRORE:
        value *= 1e7
    elif unit in _LAKH:
        value *= 1e5
    elif unit in _THOUSAND:
        value *= 1e3
    elif unit == "" and value < 1000:
        value *= 1e5
    return value if value > 0 else None


# --------------------------------------------------------------------------
# KNN home finder
# --------------------------------------------------------------------------
_KNN_COLS = ["price_inr", "GrLivArea", "OverallQual", "FullBath", "GarageCars"]
_KNN_WEIGHTS = np.array([2.0, 1.0, 1.0, 0.5, 0.5])  # price counts the most


def recommend(df: pd.DataFrame, budget_inr: float, rate: float, prefs: dict, k: int = 5):
    """K nearest houses to the budget and preferences. Returns (table, note)."""
    pool = df.copy()
    pool["price_inr"] = pool[TARGET] * rate
    pool["house"] = pool["Id"] if "Id" in pool.columns else np.arange(1, len(pool) + 1)

    scaler = StandardScaler().fit(pool[_KNN_COLS])

    note = None
    candidates = pool[pool["price_inr"] <= budget_inr]
    if len(candidates) >= k:
        search = candidates
    else:
        search = pool
        note = (
            f"Only {len(candidates)} house(s) in the data fit this budget, "
            "so the closest matches overall are shown."
        )

    k = max(1, min(k, len(search)))
    query = pd.DataFrame(
        [
            [
                budget_inr,
                prefs["GrLivArea"],
                prefs["OverallQual"],
                prefs["FullBath"],
                prefs["GarageCars"],
            ]
        ],
        columns=_KNN_COLS,
    )
    nn = NearestNeighbors(n_neighbors=k).fit(scaler.transform(search[_KNN_COLS]) * _KNN_WEIGHTS)
    _, idx = nn.kneighbors(scaler.transform(query) * _KNN_WEIGHTS)

    result = search.iloc[idx[0]].copy()
    result["within_budget"] = result["price_inr"] <= budget_inr
    cols = ["house", "price_inr", "GrLivArea", "OverallQual", "FullBath", "GarageCars", "YearBuilt", "within_budget"]
    return result[cols].reset_index(drop=True), note


# --------------------------------------------------------------------------
# EMI
# --------------------------------------------------------------------------
def emi(principal: float, annual_rate_pct: float, years: float) -> float:
    n = int(round(years * 12))
    if n <= 0:
        raise ValueError("Tenure must be at least one month.")
    if principal <= 0:
        return 0.0
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return principal / n
    f = (1 + r) ** n
    return principal * r * f / (f - 1)


def amortization(principal: float, annual_rate_pct: float, years: float) -> pd.DataFrame:
    """Month-by-month schedule: interest, principal paid and closing balance."""
    n = int(round(years * 12))
    payment = emi(principal, annual_rate_pct, years)
    r = annual_rate_pct / 12 / 100
    balance = principal
    rows = []
    for month in range(1, n + 1):
        interest = balance * r
        principal_paid = payment - interest
        balance = max(balance - principal_paid, 0.0)
        rows.append((month, payment, interest, principal_paid, balance))
    return pd.DataFrame(
        rows, columns=["month", "emi", "interest", "principal_paid", "balance"]
    )


# --------------------------------------------------------------------------
# Payment ledger (SQLite)
# --------------------------------------------------------------------------
_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_label TEXT NOT NULL,
    due_date TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount > 0),
    status TEXT NOT NULL DEFAULT 'Pending' CHECK (status IN ('Paid', 'Pending')),
    paid_on TEXT,
    note TEXT DEFAULT ''
)
"""


@contextmanager
def _db(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_LEDGER_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _iso(d) -> str:
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


def ledger_add(db_path, label: str, due, amount: float, note: str = "") -> int:
    label = (label or "").strip()
    if not label:
        raise ValueError("Please enter a property name or label.")
    if amount is None or float(amount) <= 0:
        raise ValueError("Amount must be greater than zero.")
    with _db(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO payments (property_label, due_date, amount, status, note) "
            "VALUES (?, ?, ?, 'Pending', ?)",
            (label, _iso(due), float(amount), note or ""),
        )
        return int(cur.lastrowid)


def ledger_list(db_path, status: str | None = None) -> pd.DataFrame:
    query = (
        "SELECT id, property_label, due_date, amount, status, paid_on, note "
        "FROM payments"
    )
    params: tuple = ()
    if status in ("Paid", "Pending"):
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY due_date, id"
    with _db(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    today = date.today().isoformat()
    df["overdue"] = (df["status"] == "Pending") & (df["due_date"] < today)
    return df


def ledger_set_status(db_path, entry_id: int, status: str, paid_on=None) -> bool:
    if status not in ("Paid", "Pending"):
        raise ValueError("Status must be 'Paid' or 'Pending'.")
    paid = _iso(paid_on or date.today()) if status == "Paid" else None
    with _db(db_path) as conn:
        cur = conn.execute(
            "UPDATE payments SET status = ?, paid_on = ? WHERE id = ?",
            (status, paid, int(entry_id)),
        )
        return cur.rowcount > 0


def ledger_delete(db_path, entry_id: int) -> bool:
    with _db(db_path) as conn:
        cur = conn.execute("DELETE FROM payments WHERE id = ?", (int(entry_id),))
        return cur.rowcount > 0


def ledger_summary(df: pd.DataFrame) -> dict:
    paid = df[df["status"] == "Paid"]
    pending = df[df["status"] == "Pending"]
    return {
        "total_paid": float(paid["amount"].sum()),
        "total_pending": float(pending["amount"].sum()),
        "n_paid": int(len(paid)),
        "n_pending": int(len(pending)),
        "n_overdue": int(df["overdue"].sum()) if len(df) else 0,
    }


MAX_SCHEDULE_MONTHS = 360


def ledger_add_emi_schedule(db_path, label: str, emi_amount: float, months: int, start) -> int:
    """Add one Pending ledger row per monthly EMI. Returns rows added."""
    label = (label or "").strip()
    if not label:
        raise ValueError("Please enter a property name or label.")
    if emi_amount is None or float(emi_amount) <= 0:
        raise ValueError("EMI must be greater than zero.")
    months = int(min(max(months, 1), MAX_SCHEDULE_MONTHS))
    start_ts = pd.Timestamp(start)
    rows = [
        (
            label,
            (start_ts + pd.DateOffset(months=i)).date().isoformat(),
            float(emi_amount),
            f"EMI {i + 1} of {months}",
        )
        for i in range(months)
    ]
    with _db(db_path) as conn:
        conn.executemany(
            "INSERT INTO payments (property_label, due_date, amount, status, note) "
            "VALUES (?, ?, ?, 'Pending', ?)",
            rows,
        )
    return len(rows)