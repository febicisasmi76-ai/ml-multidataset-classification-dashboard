import pandas as pd
import numpy as np
import streamlit as st

HEALTH_LINK = "https://github.com/advikmaniar/ML-Healthcare-Web-App/tree/main/Data"
ENV_LINK = "https://github.com/ryanjiroo/Forecasting-Kualitas-Udara-Jakarta/tree/main/data"

def _detect_dataset(df: pd.DataFrame) -> str:
    cols = set([c.lower() for c in df.columns])
    if "diagnosis" in cols:
        return "health"
    if "categori" in cols or "kategori" in cols or "ispu" in cols or "pm10" in cols:
        return "environment"
    return "unknown"

def _prep_health(df: pd.DataFrame) -> dict:
    # diagnosis: 'M'/'B' -> 1/0
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "diagnosis" not in df.columns:
        raise ValueError("Kolom 'diagnosis' tidak ditemukan untuk dataset kesehatan.")

    # map diagnosis
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0}).fillna(df["diagnosis"])

    # drop id if exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # pastikan numerik
    for c in df.columns:
        if c != "diagnosis":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].astype(int)

    meta = {
        "dataset_type": "health",
        "target_col": "diagnosis",
        "positive_label": "Malignant (Ganas)",
        "negative_label": "Benign (Jinak)",
        "dataset_link": HEALTH_LINK
    }
    return {"df": df, "X": X, "y": y, "meta": meta}

def _prep_environment(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # target bisa "categori" (sesuai file yang umum)
    target_candidates = [c for c in df.columns if c.lower() in ["categori", "kategori", "category", "label"]]
    if not target_candidates:
        raise ValueError("Kolom kategori (mis. 'categori') tidak ditemukan untuk dataset lingkungan.")
    target_col = target_candidates[0]

    # convert label -> binary AMAN
    safe_labels = {"BAIK", "SEDANG"}  # kamu bisa sesuaikan jika dosen punya definisi lain
    unsafe_labels = {"TIDAK SEHAT", "SANGAT TIDAK SEHAT", "BERBAHAYA"}

    df[target_col] = df[target_col].astype(str).str.upper().str.strip()

    def to_binary(label: str) -> int:
        if label in safe_labels:
            return 1  # AMAN
        if label in unsafe_labels:
            return 0  # TIDAK AMAN
        # kalau label lain/unknown -> anggap aman? lebih aman: jadikan NaN lalu drop
        return np.nan

    df["target_aman"] = df[target_col].apply(to_binary)
    df = df.dropna(subset=["target_aman"])

    # pilih fitur numerik utama
    # biasanya: pm10, pm25, so2, co, o3, no2, max
    numeric_candidates = ["pm10", "pm25", "so2", "co", "o3", "no2", "max"]
    cols_lower = {c.lower(): c for c in df.columns}
    feature_cols = [cols_lower[c] for c in numeric_candidates if c in cols_lower]

    # tambah 'stasiun' sebagai kategori bila ada
    station_col = None
    for c in df.columns:
        if c.lower() in ["stasiun", "station"]:
            station_col = c
            break

    use_cols = feature_cols + ([station_col] if station_col else [])
    sub = df[use_cols + ["target_aman"]].copy()

    # numeric convert + impute median
    for c in feature_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub[c] = sub[c].fillna(sub[c].median())

    # one-hot for station
    if station_col:
        sub[station_col] = sub[station_col].astype(str)
        sub = pd.get_dummies(sub, columns=[station_col], drop_first=True)

    X = sub.drop(columns=["target_aman"])
    y = sub["target_aman"].astype(int)

    meta = {
        "dataset_type": "environment",
        "target_col": "target_aman",
        "positive_label": "AMAN",
        "negative_label": "TIDAK AMAN",
        "dataset_link": ENV_LINK,
        "original_label_col": target_col
    }
    return {"df": sub, "X": X, "y": y, "meta": meta}

@st.cache_data
def load_and_prepare(uploaded_file, dataset_mode: str):
    if uploaded_file is None:
        return None

    df = pd.read_csv(uploaded_file)

    # tentukan tipe dataset
    if dataset_mode == "Kesehatan (Breast Cancer)":
        dtype = "health"
    elif dataset_mode == "Lingkungan (ISPU Udara)":
        dtype = "environment"
    else:
        dtype = _detect_dataset(df)

    if dtype == "health":
        return _prep_health(df)

    if dtype == "environment":
        return _prep_environment(df)

    return {"error": "Dataset tidak dikenali. Pastikan kolom 'diagnosis' (kesehatan) atau 'categori' (lingkungan) ada."}
