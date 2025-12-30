import streamlit as st

def contact_page():
    st.title("👤 Contact & Final Summary")

    # =========================
    # RINGKASAN AKHIR
    # =========================
    st.markdown("""
    <div class="card card-primary">
      <h3>📌 Ringkasan Akhir Dashboard</h3>
      <p>
        Dashboard ini merupakan <b>sistem klasifikasi multi-dataset</b>
        yang dirancang untuk menangani dua domain berbeda:
      </p>
      <ul>
        <li><b>Kesehatan</b> — klasifikasi kanker payudara</li>
        <li><b>Lingkungan</b> — klasifikasi kualitas udara (ISPU)</li>
      </ul>

      <p><b>Fitur Utama:</b></p>
      <ul>
        <li>Upload dataset CSV</li>
        <li>Statistik deskriptif (mean, median, Q1, Q3)</li>
        <li>Visualisasi interaktif + interpretasi</li>
        <li>Komparasi 5 algoritma klasifikasi</li>
        <li>Pemilihan model terbaik</li>
        <li>Prediksi data baru + rekomendasi (Decision Support System)</li>
      </ul>

      <p><b>Link Dataset:</b></p>
      <ul>
        <li>
          Dataset Kesehatan (Breast Cancer):
          <a href="https://github.com/advikmaniar/ML-Healthcare-Web-App/tree/main/Data" target="_blank">
            Klik di sini
          </a>
        </li>
        <li>
          Dataset Lingkungan (ISPU Jakarta):
          <a href="https://github.com/ryanjiroo/Forecasting-Kualitas-Udara-Jakarta/tree/main/data" target="_blank">
            Klik di sini
          </a>
        </li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =========================
    # PROFIL PENGEMBANG
    # =========================
    st.markdown("""
    <div class="card card-success">
      <h3>👩‍💻 Profil Pengembang</h3>

      <p><b>Nama:</b> Febi Anggun Lestari</p>

      <p>
        <b>Email:</b><br>
        <a href="mailto:vebbyanggunlestari@gmail.com">
          vebbyanggunlestari@gmail.com
        </a>
      </p>

      <p>
        <b>GitHub:</b><br>
        <a href="https://github.com/febicisasmi76-ai/porto-unimus1" target="_blank">
          https://github.com/febicisasmi76-ai/porto-unimus1
        </a>
      </p>

      <p>
        <b>LinkedIn:</b><br>
        <a href="https://www.linkedin.com/in/febi-anggun-lestari-8026422b0/" target="_blank">
          https://www.linkedin.com/in/febi-anggun-lestari-8026422b0/
        </a>
      </p>

      <p style="margin-top:12px; font-size:14px; opacity:.8;">
        Dashboard ini dibuat untuk memenuhi tugas akhir mata kuliah
        <b>Machine Learning</b>.
      </p>
    </div>
    """, unsafe_allow_html=True)
