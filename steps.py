import streamlit as st

def steps_page():
    st.header("🧭 Steps / Metodologi (Runtut & Lengkap)")

    st.markdown(
        """
<div class="card cardTopBlue softGlowBlue">
  <h3>✅ Alur Umum (Berlaku untuk 2 Dataset)</h3>
  <div class="smallMuted">
    <ol>
      <li><b>Upload Data</b> (CSV) melalui sidebar</li>
      <li><b>Auto-detect dataset</b> atau pilih mode (kesehatan/lingkungan)</li>
      <li><b>Data cleaning</b> (missing value, tipe data, encoding bila perlu)</li>
      <li><b>Statistik deskriptif</b> (count, mean, median, Q1, Q3, std, min, max)</li>
      <li><b>EDA & Visualisasi</b> (distribusi, hubungan fitur, korelasi)</li>
      <li><b>Split data</b> (train-test)</li>
      <li><b>Standardisasi</b> (scaler) agar model stabil</li>
      <li><b>Training & Evaluasi</b> beberapa algoritma</li>
      <li><b>Model terbaik</b> dipilih dengan metrik (utamanya F1-score)</li>
      <li><b>Prediction</b> data baru + rekomendasi</li>
    </ol>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### 🧠 Alur Tiap Algoritma (Klik untuk lihat detail)")

    with st.expander("1) Logistic Regression — langkah kerja"):
        st.markdown(
            """
- Tentukan fungsi sigmoid untuk memetakan probabilitas kelas  
- Optimasi parameter (gradient descent / solver)  
- Model output: probabilitas → threshold (umumnya 0.5) → kelas  
- Kelebihan: interpretasi jelas, baseline kuat  
- Kelemahan: kurang menangkap pola non-linear kompleks  
"""
        )

    with st.expander("2) KNN — langkah kerja"):
        st.markdown(
            """
- Tentukan nilai **k** (jumlah tetangga)  
- Hitung jarak data baru ke data training (mis. Euclidean)  
- Ambil k tetangga terdekat → voting mayoritas  
- Kelebihan: sederhana, efektif pada pola lokal  
- Kelemahan: sensitif skala fitur (wajib scaling), lambat jika data besar  
"""
        )

    with st.expander("3) SVM — langkah kerja"):
        st.markdown(
            """
- Cari hyperplane pemisah terbaik antar kelas  
- Gunakan margin maksimum untuk generalisasi  
- Jika non-linear: gunakan kernel (RBF/Poly)  
- Kelebihan: kuat untuk klasifikasi  
- Kelemahan: butuh tuning kernel/parameter, bisa berat komputasi  
"""
        )

    with st.expander("4) Decision Tree — langkah kerja"):
        st.markdown(
            """
- Memilih fitur terbaik untuk split (Gini/Entropy)  
- Membentuk node hingga kondisi berhenti (depth/min_samples)  
- Kelebihan: mudah dipahami, menangkap non-linear  
- Kelemahan: mudah overfitting jika tidak dibatasi  
"""
        )

    with st.expander("5) Random Forest — langkah kerja"):
        st.markdown(
            """
- Membuat banyak decision tree (bagging)  
- Tiap tree dilatih dari sampel acak data + subset fitur acak  
- Prediksi akhir = voting mayoritas  
- Kelebihan: stabil, biasanya performa tinggi  
- Kelemahan: interpretasi lebih sulit, butuh komputasi lebih  
"""
        )

    with st.expander("6) Gradient Boosting (opsional untuk lingkungan) — langkah kerja"):
        st.markdown(
            """
- Melatih tree kecil bertahap (sequential)  
- Setiap model baru memperbaiki error model sebelumnya  
- Kelebihan: sangat kuat untuk pola kompleks/noisy (sering bagus untuk data lingkungan)  
- Kelemahan: sensitif parameter, bisa overfitting jika tidak dituning  
"""
        )
