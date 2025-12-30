import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_and_prepare


# =========================================================
# PREDICTION PAGE (BEST MODEL ONLY)
# =========================================================
def prediction_page():
    st.header("🔮 Prediction & Recommendation (Best Model)")

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

        # probabilitas (jika tersedia)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(input_df)[0][pred]) * 100

        label = meta["positive_label"] if pred == 1 else meta["negative_label"]

        # =================================================
        # RESULT CARD
        # =================================================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📌 Hasil Prediksi")

        if pred == 1:
            st.markdown(
                f"""
<div class="card" style="background:linear-gradient(135deg,#16A34A,#22C55E);color:white;">
  <h2 style="margin:0;">✅ {label}</h2>
  <div style="margin-top:8px;font-size:18px;">
    Model mendeteksi kondisi <b>positif</b>.
  </div>
  <div style="margin-top:8px;">
    <b>Confidence:</b> {prob:.2f}%
  </div>
</div>
""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
<div class="card" style="background:linear-gradient(135deg,#DC2626,#EF4444);color:white;">
  <h2 style="margin:0;">⚠️ {label}</h2>
  <div style="margin-top:8px;font-size:18px;">
    Model mendeteksi kondisi <b>negatif / berisiko</b>.
  </div>
  <div style="margin-top:8px;">
    <b>Confidence:</b> {prob:.2f}%
  </div>
</div>
""",
                unsafe_allow_html=True
            )

        # =================================================
        # REKOMENDASI TINDAKAN (DSS)
        # =================================================
        st.markdown("<br>", unsafe_allow_html=True)

        if meta["dataset_type"] == "health":
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

        else:
            st.markdown("### 🌿 Rekomendasi Tindakan (Lingkungan)")

            if pred == 1:
                st.markdown("""
<div class="card">
  <ul>
    <li>Kualitas udara relatif aman untuk aktivitas harian.</li>
    <li>Tetap waspada bagi kelompok sensitif (asma, lansia, anak-anak).</li>
    <li>Jaga kualitas udara dalam ruangan.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div class="card">
  <ul>
    <li>Kurangi aktivitas luar ruangan.</li>
    <li>Gunakan masker jika harus beraktivitas di luar.</li>
    <li>Kelompok rentan sebaiknya tetap berada di dalam ruangan.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="card" style="background:#F1F5F9;">
⚠️ <b>Catatan:</b> Rekomendasi ini bersifat panduan umum
berdasarkan hasil prediksi model.
</div>
""", unsafe_allow_html=True)
