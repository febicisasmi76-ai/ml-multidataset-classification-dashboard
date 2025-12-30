import streamlit as st
from data_loader import HEALTH_LINK, ENV_LINK

def about_page():
    st.header("📘 About the Project & Dataset")

    st.markdown(
        """
<div class="card cardTopBlue softGlowBlue">
  <h3>🧩 Gambaran Umum</h3>
  <div class="smallMuted">
    Dashboard ini adalah <b>framework klasifikasi</b> yang dapat digunakan pada dua domain:
    <b>kesehatan</b> dan <b>lingkungan</b>. Sistem menyediakan:
    <ul>
      <li>Upload dataset</li>
      <li>Statistik deskriptif (mean, median, Q1, Q3, dst.)</li>
      <li>Visualisasi interaktif + interpretasi</li>
      <li>Komparasi beberapa algoritma</li>
      <li>Prediksi data baru + rekomendasi (Decision Support)</li>
    </ul>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
<div class="card cardTopPurple softGlowPurple">
  <h3>🩺 Dataset Kesehatan (Breast Cancer)</h3>
  <div class="smallMuted">
    <b>Tujuan:</b> Mengklasifikasikan tumor menjadi:
    <ul>
      <li><b>Benign (Jinak)</b></li>
      <li><b>Malignant (Ganas)</b></li>
    </ul>
    <b>Manfaat:</b> Mendukung deteksi dini dan sebagai <i>decision support</i>
    (alat bantu) bagi skrining awal.
    <br><br>
    <b>Karakteristik:</b> fitur numerik hasil ekstraksi citra sel tumor (radius, area, texture, dll).
    <br><br>
    <b>Link Dataset:</b> {HEALTH_LINK}
  </div>
</div>
""",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
<div class="card cardTopGreen softGlowGreen">
  <h3>🌍 Dataset Lingkungan (Kualitas Udara / ISPU)</h3>
  <div class="smallMuted">
    <b>Tujuan:</b> Mengklasifikasikan kualitas udara menjadi:
    <ul>
      <li><b>AMAN</b> (BAIK/SEDANG)</li>
      <li><b>TIDAK AMAN</b> (TIDAK SEHAT, dst.)</li>
    </ul>
    <b>Manfaat:</b> Mendukung peringatan dini dan rekomendasi aktivitas harian masyarakat.
    <br><br>
    <b>Karakteristik:</b> parameter polutan (PM10, PM2.5, SO2, CO, O3, NO2, max)
    + (opsional) stasiun.
    <br><br>
    <b>Link Dataset:</b> {ENV_LINK}
  </div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="card">
  <h3>🧠 Decision Support System (DSS) itu apa?</h3>
  <div class="smallMuted">
    DSS adalah sistem yang <b>membantu pengambilan keputusan</b> dengan cara:
    <ol>
      <li>Memberikan prediksi berbasis model</li>
      <li>Menunjukkan interpretasi/insight dari data</li>
      <li>Memberikan rekomendasi tindakan sesuai hasil</li>
    </ol>
    <b>Catatan penting:</b> hasil prediksi bersifat pendukung (bukan pengganti keputusan ahli/dokter).
  </div>
</div>
""",
        unsafe_allow_html=True
    )
