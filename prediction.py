import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_and_prepare


# =========================================================
# PREDICTION PAGE (BEST MODEL ONLY)
# =========================================================
def prediction_page():
    # =====================================================
    # JUDUL APLIKASI
    # =====================================================
    st.header("🩺 Aplikasi Prediksi Kanker Payudara")

    st.markdown("""
<div class="card" style="background:#F8FAFC;border-left:6px solid #DC2626;">
  <h4>Berbasis Machine Learning</h4>
  <div class="smallMuted">
    Aplikasi ini memprediksi kondisi <b>jinak (benign)</b> atau
    <b>ganas (malignant)</b> pada kanker payudara.
    <br><br>
    <i>Pendekatan klasifikasi yang digunakan bersifat umum dan
    dapat dikembangkan untuk dataset lain.</i>
  </div>
</div>
""", unsafe_allow_html=True)

    uploaded = st.session_state.get("uploaded_file")
    mode = st.session_state.get("dataset_mode", "Auto Detect")
    trained_pack = st.session_state.get("trained_pack")

    pack = load_and_prepare(uploaded, mode)

    if uploaded is None:
        st.warning("Silakan upload dataset CSV di sidebar terlebih dahulu.")
        return

    if pack is None or "error" in pack:
        st.error(pack.get("error", "Gagal memproses dataset."))
        return

    # =====================================================
    # KUNCI DATASET: HANYA KESEHATAN
    # =====================================================
    if pack["meta"]["dataset_type"] != "health":
        st.error(
            "Halaman prediksi ini khusus untuk dataset kesehatan "
            "(Kanker Payudara). Dataset lain digunakan pada tahap modeling."
        )
        return

    if trained_pack is None:
        st.warning("Silakan lakukan proses Modeling terlebih dahulu untuk menentukan model terbaik.")
        return

    X = pack["X"]
    meta = pack["meta"]

    best_model_name = trained_pack["best_model_name"]
    model = trained_pack["models"][best_model_name]

    # =====================================================
    # INFO MODEL
    # =====================================================
    st.markdown(
        f"""
<div class="card cardTopGreen softGlowGreen">
  <h3>🏆 Model yang Digunakan</h3>
  <div class="smallMuted">
    Prediction menggunakan <b>model terbaik</b> hasil tahap Modeling.<br>
    <b>Algoritma:</b> {best_model_name}
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    # =====================================================
    # INPUT DATA
    # =====================================================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📝 Input Data Baru")

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(X.columns):
        with cols[i % 3]:
            default_val = float(X[feature].mean())
            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=default_val
            )

    # =====================================================
    # PREDICT
    # =====================================================
    if st.button("🔍 Jalankan Prediksi", use_container_width=True):
        input_df = pd.DataFrame([input_data])

        # prediksi kelas
        pred = int(model.predict(input_df)[0])

        # probabilitas (AMAN UNTUK SEMUA MODEL)
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)
                prob = float(proba[0][pred]) * 100
            except Exception:
                prob = None

        label = meta["positive_label"] if pred == 1 else meta["negative_label"]

        # =================================================
        # RESULT CARD
        # =================================================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📌 Hasil Prediksi")

        confidence_text = f"{prob:.2f}%" if prob is not None else "N/A"

        if pred == 1:
            # GANAS → MERAH
            st.markdown(
                f"""
<div class="card" style="background:linear-gradient(135deg,#DC2626,#EF4444);color:white;">
  <h2 style="margin:0;">⚠️ {label}</h2>
  <div style="margin-top:8px;font-size:18px;">
    Model memprediksi kondisi <b>GANAS</b> dan berisiko tinggi.
  </div>
  <div style="margin-top:8px;">
    <b>Confidence:</b> {confidence_text}
  </div>
</div>
""",
                unsafe_allow_html=True
            )
        else:
            # JINAK → HIJAU
            st.markdown(
                f"""
<div class="card" style="background:linear-gradient(135deg,#16A34A,#22C55E);color:white;">
  <h2 style="margin:0;">✅ {label}</h2>
  <div style="margin-top:8px;font-size:18px;">
    Model memprediksi kondisi <b>JINAK</b> dan relatif aman.
  </div>
  <div style="margin-top:8px;">
    <b>Confidence:</b> {confidence_text}
  </div>
</div>
""",
                unsafe_allow_html=True
            )

        # =================================================
        # REKOMENDASI TINDAKAN (DSS)
        # =================================================
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🩺 Rekomendasi Tindakan (Kesehatan)")

        if pred == 1:
            st.markdown("""
<div class="card">
  <ul>
    <li>Segera lakukan konsultasi dengan dokter atau tenaga medis.</li>
    <li>Lakukan pemeriksaan lanjutan seperti USG, mammografi, atau biopsi.</li>
    <li>Prioritaskan penanganan dini untuk mengurangi risiko komplikasi.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="card">
  <ul>
    <li>Tetap lakukan pemeriksaan rutin secara berkala.</li>
    <li>Jaga pola hidup sehat dan waspadai perubahan gejala.</li>
    <li>Gunakan hasil ini sebagai pendukung, bukan diagnosis final.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="card" style="background:#F1F5F9;">
⚠️ <b>Catatan:</b> Sistem ini merupakan <b>Decision Support System</b> dan
tidak menggantikan keputusan medis profesional.
</div>
""", unsafe_allow_html=True)
