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
- Menginisialisasi parameter bobot dan bias  
- Menggunakan fungsi sigmoid untuk memetakan nilai ke probabilitas  
- Menghitung loss (log-loss)  
- Melakukan optimasi parameter menggunakan metode iteratif  
- Menghasilkan probabilitas kelas dan menentukan kelas berdasarkan threshold  
"""
        )

    with st.expander("2) KNN — langkah kerja"):
        st.markdown(
            """
- Menentukan nilai **k** (jumlah tetangga terdekat)  
- Menghitung jarak antara data uji dan data latih  
- Memilih k data dengan jarak terdekat  
- Menentukan kelas berdasarkan voting mayoritas tetangga  
"""
        )

    with st.expander("3) SVM — langkah kerja"):
        st.markdown(
            """
- Menentukan dan mencari nilai bobot (w1, w2, ..., wn) serta bias  
- Melakukan optimasi bobot untuk memaksimalkan margin  
- Menentukan hyperplane pemisah terbaik antar kelas  
- Menggunakan margin sebagai dasar generalisasi model  
"""
        )

    with st.expander("4) Decision Tree — langkah kerja"):
        st.markdown(
            """
- Memilih fitur terbaik sebagai pemisah data  
- Melakukan proses split berdasarkan kriteria tertentu  
- Membentuk node dan cabang hingga kondisi berhenti tercapai  
- Menghasilkan struktur pohon keputusan untuk klasifikasi  
"""
        )

    with st.expander("5) Random Forest — langkah kerja"):
        st.markdown(
            """
- Membuat beberapa subset data secara acak  
- Melatih banyak decision tree secara independen  
- Setiap tree melakukan prediksi kelas  
- Menentukan hasil akhir berdasarkan voting mayoritas  
"""
        )

    with st.expander("6) Gradient Boosting — langkah kerja"):
        st.markdown(
            """
- Melatih model secara bertahap (sequential)  
- Setiap model baru mempelajari kesalahan model sebelumnya  
- Menggabungkan seluruh model untuk menghasilkan prediksi akhir  
"""
        )
