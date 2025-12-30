import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_and_prepare

def _descriptive_stats(df: pd.DataFrame):
    # statistik deskriptif yang diminta dosen
    num = df.select_dtypes(include="number")
    if num.empty:
        return None
    out = pd.DataFrame({
        "count": num.count(),
        "mean": num.mean(),
        "std": num.std(),
        "min": num.min(),
        "Q1": num.quantile(0.25),
        "median": num.median(),
        "Q3": num.quantile(0.75),
        "max": num.max(),
    }).round(3)
    return out

def visualization_page():
    st.header("📊 Visualization & Descriptive Statistics")

    uploaded = st.session_state.get("uploaded_file")
    mode = st.session_state.get("dataset_mode", "Auto Detect")
    pack = load_and_prepare(uploaded, mode)

    if uploaded is None:
        st.warning("Silakan upload dataset CSV di sidebar terlebih dahulu.")
        return

    if pack is None or "error" in pack:
        st.error(pack.get("error", "Gagal memproses dataset."))
        return

    df = pack["df"]
    X = pack["X"]
    y = pack["y"]
    meta = pack["meta"]

    # KPI
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", len(df))
    c2.metric(meta["positive_label"], int((y == 1).sum()))
    c3.metric(meta["negative_label"], int((y == 0).sum()))

    st.markdown("<hr>", unsafe_allow_html=True)

    # FILTER
    left, right = st.columns([2, 1])
    with left:
        st.subheader("🎛 Filter Interaktif")
    with right:
        st.markdown(
            f"""
<div class="card">
  <b>Mode:</b> {meta["dataset_type"].upper()}<br>
  <b>Target:</b> {meta["target_col"]}
</div>
""",
            unsafe_allow_html=True
        )

    # pilih feature untuk plot (biar adaptif)
    feat_cols = list(X.columns)
    if len(feat_cols) < 2:
        st.error("Fitur terlalu sedikit untuk visualisasi.")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        x_col = st.selectbox("Feature X (Histogram/Scatter)", feat_cols, index=0)
    with f2:
        y_col = st.selectbox("Feature Y (Scatter/Box)", feat_cols, index=1 if len(feat_cols) > 1 else 0)
    with f3:
        sample_n = st.slider("Sample untuk scatter (cepat)", 300, min(2000, len(df)), min(800, len(df)))

    plot_df = df.copy()
    plot_df["_target"] = y.values

    # =========================
    # DESCRIPTIVE STATISTICS
    # =========================
    st.subheader("📌 Statistik Deskriptif (Mean, Median, Q1, Q3, dst.)")
    stats = _descriptive_stats(df)
    if stats is not None:
        st.dataframe(stats, use_container_width=True)
        with st.expander("🧠 Interpretasi Statistik Deskriptif + Rekomendasi"):
            st.markdown(
                """
- **Mean & Median** membantu melihat pusat data dan indikasi skew (jika jauh berbeda).  
- **Q1 & Q3** menunjukkan sebaran data (IQR = Q3−Q1), berguna untuk deteksi outlier.  
- **Std** menunjukkan variasi: semakin besar, semakin beragam nilai fitur.  

**Rekomendasi:**
- Jika variasi fitur sangat berbeda antar kolom, gunakan **StandardScaler** (sudah dilakukan di modeling).
- Jika ditemukan outlier ekstrem, pertimbangkan treatment (winsorize / robust scaling).
"""
            )
    else:
        st.info("Tidak ada kolom numerik untuk statistik deskriptif.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # =========================
    # 2-COLUMN CHART LAYOUT
    # =========================
    colL, colR = st.columns(2)

    with colL:
        st.subheader("1) Distribusi Kelas (Class Balance)")
        fig = px.histogram(plot_df, x="_target", text_auto=True)
        fig.update_layout(xaxis_title="Target", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📖 Interpretasi + Rekomendasi (Distribusi Kelas)"):
            st.markdown(
                f"""
**Interpretasi:**  
Grafik ini menunjukkan jumlah data untuk kelas **{meta['negative_label']}** vs **{meta['positive_label']}**.

**Rekomendasi:**  
- Jika kelas tidak seimbang (imbalance), gunakan:  
  - class_weight, threshold tuning, atau teknik resampling.  
- Fokus metrik: **F1-score / Recall** lebih penting daripada akurasi saja (terutama domain kesehatan).
"""
            )

        st.subheader("2) Histogram Feature (klik legend untuk hide/show)")
        fig = px.histogram(plot_df, x=x_col, color="_target", barmode="overlay", opacity=0.6)
        fig.update_layout(title=f"Distribusi {x_col}", xaxis_title=x_col, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📖 Interpretasi + Rekomendasi (Histogram)"):
            st.markdown(
                """
**Interpretasi:**  
Histogram membandingkan sebaran fitur pada masing-masing kelas. Jika bentuk distribusi antar kelas berbeda jauh,
fitur tersebut cenderung **informatif** untuk klasifikasi.

**Rekomendasi:**  
- Pilih fitur yang memberikan pemisahan jelas antar kelas.
- Jika banyak overlap, model non-linear (RF/Boosting) biasanya lebih baik.
"""
            )

        st.subheader("3) Boxplot Feature")
        fig = px.box(plot_df, x="_target", y=y_col, points="outliers")
        fig.update_layout(title=f"Boxplot {y_col} per Kelas", xaxis_title="Target", yaxis_title=y_col)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📖 Interpretasi + Rekomendasi (Boxplot)"):
            st.markdown(
                """
**Interpretasi:**  
Boxplot menunjukkan median, Q1, Q3 dan outlier per kelas.
Jika median dan IQR antar kelas berbeda jelas → fitur punya daya pisah lebih baik.

**Rekomendasi:**  
- Jika banyak outlier, pertimbangkan RobustScaler atau pembersihan outlier.
- Boxplot membantu memilih fitur prioritas untuk modeling.
"""
            )

    with colR:
        st.subheader("4) Scatter (Interaktif + Hover)")
        sc_df = plot_df.sample(n=min(sample_n, len(plot_df)), random_state=42)
        fig = px.scatter(sc_df, x=x_col, y=y_col, color="_target", hover_data=sc_df.columns[:8])
        fig.update_layout(title=f"{x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📖 Interpretasi + Rekomendasi (Scatter)"):
            st.markdown(
                """
**Interpretasi:**  
Scatter melihat hubungan dua fitur sekaligus.
Jika titik dua kelas terlihat terpisah → kombinasi fitur tersebut kuat.

**Rekomendasi:**  
- Jika pemisahan terlihat non-linear, gunakan RF/Boosting.
- Jika pemisahan linear, Logistic Regression bisa sangat bagus.
"""
            )

        st.subheader("5) Correlation Heatmap")
        corr = pd.DataFrame(X).corr(numeric_only=True)
        fig = px.imshow(corr, aspect="auto")
        fig.update_layout(title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📖 Interpretasi + Rekomendasi (Heatmap)"):
            st.markdown(
                """
**Interpretasi:**  
Korelasi tinggi antar fitur bisa menyebabkan redundansi/multikolinearitas.

**Rekomendasi:**  
- Jika ada banyak korelasi sangat tinggi, pertimbangkan feature selection atau PCA.
- Namun model tree-based (Decision Tree / Random Forest) biasanya lebih tahan terhadap multikolinearitas.
"""
            )
