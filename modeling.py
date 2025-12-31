import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import plotly.express as px

from data_loader import load_and_prepare


# =========================================================
# MODEL REGISTRY
# =========================================================
def _get_models(dataset_type: str):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    if dataset_type == "environment":
        models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
    return models


# =========================================================
# MAIN PAGE
# =========================================================
def modeling_page():
    st.header("🤖 Modeling & Analisis Algoritma")

    uploaded = st.session_state.get("uploaded_file")
    mode = st.session_state.get("dataset_mode", "Auto Detect")
    pack = load_and_prepare(uploaded, mode)

    if uploaded is None:
        st.warning("Silakan upload dataset CSV di sidebar terlebih dahulu.")
        return
    if pack is None or "error" in pack:
        st.error(pack.get("error", "Gagal memproses dataset."))
        return

    X = pack["X"]
    y = pack["y"]
    meta = pack["meta"]

    # =====================================================
    # INFO DATASET
    # =====================================================
    st.markdown(
        f"""
<div class="card cardTopBlue softGlowBlue">
  <h3>📌 Dataset: {meta['dataset_type'].upper()}</h3>
  <div class="smallMuted">
    Target: <b>{meta['positive_label']}</b> vs <b>{meta['negative_label']}</b><br>
    Evaluasi utama menggunakan <b>F1-score</b> dan <b>ROC–AUC</b>.
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # =====================================================
    # SPLIT DATA
    # =====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = _get_models(meta["dataset_type"])

    # =====================================================
    # ANALISIS SATU MODEL
    # =====================================================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🎯 Analisis Algoritma")

    # 👉 DEFAULT = Random Forest (BIAR FEATURE IMPORTANCE LANGSUNG MUNCUL)
    model_choice = st.selectbox(
        "Pilih algoritma untuk dianalisis:",
        list(models.keys()),
        index=list(models.keys()).index("Random Forest")
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", models[model_choice])
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    # =====================================================
    # METRIK
    # =====================================================
    st.markdown(
        f"""
<div class="card cardTopPurple softGlowPurple">
  <h3>📊 Evaluasi Model: {model_choice}</h3>
  <div style="display:flex;gap:40px;flex-wrap:wrap;margin-top:14px;">
    <div><div class="smallMuted">Accuracy</div><h2>{acc:.3f}</h2></div>
    <div><div class="smallMuted">Precision</div><h2>{prec:.3f}</h2></div>
    <div><div class="smallMuted">Recall</div><h2>{rec:.3f}</h2></div>
    <div><div class="smallMuted">F1-score</div><h2>{f1:.3f}</h2></div>
    <div><div class="smallMuted">ROC–AUC</div><h2>{auc:.3f}</h2></div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # ROC CURVE
    # =====================================================
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    roc_df = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr
    })

    fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve – {model_choice} (AUC = {auc:.3f})"
    )
    fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # FEATURE IMPORTANCE (AMAN & KONSISTEN)
    # =====================================================
    if model_choice in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
        st.subheader("📌 Feature Importance")

        importances = pipe.named_steps["model"].feature_importances_
        fi_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        fig = px.bar(
            fi_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Feature Importance (Tertinggi → Terendah)"
        )
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🧠 Interpretasi Feature Importance"):
            st.markdown("""
- Fitur dengan nilai importance tertinggi memiliki pengaruh paling besar terhadap prediksi.
- Informasi ini membantu interpretasi model pada domain kesehatan maupun lingkungan.
- Feature importance juga dapat digunakan untuk feature selection pada pengembangan lanjutan.
""")
    else:
        st.info(
            "Feature Importance hanya tersedia untuk algoritma berbasis tree "
            "(Decision Tree, Random Forest, dan Gradient Boosting)."
        )

    # =====================================================
    # KOMPARASI SEMUA MODEL
    # =====================================================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📊 Tabel Perbandingan Semua Model")

    results = []
    trained_models = {}

    for name, mdl in models.items():
        p = Pipeline([("scaler", StandardScaler()), ("model", mdl)])
        p.fit(X_train, y_train)

        pr = p.predict(X_test)
        pr_proba = p.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pr),
            "Precision": precision_score(y_test, pr, zero_division=0),
            "Recall": recall_score(y_test, pr, zero_division=0),
            "F1": f1_score(y_test, pr, zero_division=0),
            "AUC": roc_auc_score(y_test, pr_proba),
        })

        trained_models[name] = p

    result_df = pd.DataFrame(results)

    # =====================================================
    # PRIORITY TIE-BREAKER (BIAR TERPILIH 1 MODEL)
    # =====================================================
    priority_order = {
        "Random Forest": 1,
        "Gradient Boosting": 2,
        "Decision Tree": 3,
        "SVM": 4,
        "Logistic Regression": 5,
        "KNN": 6
    }

    result_df["Priority"] = result_df["Model"].map(priority_order)

    result_df = result_df.sort_values(
        by=["F1", "AUC", "Priority"],
        ascending=[False, False, True]
    )

    st.dataframe(
        result_df.drop(columns=["Priority"]).style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
            "AUC": "{:.3f}",
        }),
        use_container_width=True
    )

    # =====================================================
    # MODEL TERBAIK (FINAL)
    # =====================================================
    best = result_df.iloc[0]

    st.markdown(
        f"""
<div class="card cardTopGreen softGlowGreen">
  <h3>🏆 Model Terbaik</h3>
  <div class="smallMuted">
    Model <b>{best['Model']}</b> dipilih sebagai model terbaik karena memiliki
    nilai <b>F1-score ({best['F1']:.3f})</b> dan <b>ROC–AUC ({best['AUC']:.3f})</b>
    tertinggi.  
    Jika terdapat nilai evaluasi yang sama, pemilihan model dilakukan
    berdasarkan prioritas stabilitas dan kemampuan generalisasi.
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    with st.expander("🧠 Interpretasi & Langkah Kerja Model Terbaik"):
        st.markdown(f"""
**Interpretasi:**  
Model **{best['Model']}** memberikan performa yang stabil dan seimbang antara
Precision dan Recall, sehingga layak digunakan sebagai sistem pendukung keputusan.

**Langkah Kerja Model:**
1. Dataset dibagi menjadi data latih dan data uji.
2. Fitur dinormalisasi menggunakan StandardScaler.
3. Model dilatih menggunakan data latih.
4. Evaluasi menggunakan F1-score dan ROC–AUC.
5. Model terbaik digunakan pada tahap prediksi dan rekomendasi.
""")

    # =====================================================
    # SAVE FOR PREDICTION
    # =====================================================
    st.session_state["trained_pack"] = {
        "models": trained_models,
        "best_model_name": best["Model"],
        "feature_names": list(X.columns),
        "meta": meta
    }
