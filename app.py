import streamlit as st
import streamlit.components.v1 as components

import about
import steps
import visualisasi
import modeling
import prediction
import contact

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Multi-Dataset Classification Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# GLOBAL STYLE (MODERN WEBSITE)
# ======================================
st.markdown(
    """
<style>
:root{
  --primary:#2563EB;
  --secondary:#7C3AED;
  --success:#16A34A;
  --danger:#DC2626;
  --text:#0F172A;
  --muted:#475569;
  --card:#FFFFFF;
  --cardBorder:rgba(15,23,42,.10);
}

html { scroll-behavior:smooth; }
html, body, [class*="css"] { font-size:18px !important; color:var(--text); }

h1{font-size:40px !important; margin-bottom:.2rem;}
h2{font-size:28px !important;}
h3{font-size:22px !important;}
h4{font-size:18px !important;}

.main{
  background:
    radial-gradient(1200px 600px at 10% 0%, rgba(37,99,235,0.12), transparent 60%),
    radial-gradient(900px 500px at 90% 10%, rgba(124,58,237,0.12), transparent 55%),
    radial-gradient(1100px 650px at 50% 100%, rgba(22,163,74,0.10), transparent 60%);
}

.card{
  padding:18px 18px;
  border-radius:18px;
  background:rgba(255,255,255,.96);
  border:1px solid var(--cardBorder);
  box-shadow:0 10px 22px rgba(2,6,23,.08);
  transition:all .25s ease;
}
.card:hover{
  transform:translateY(-4px);
  box-shadow:0 18px 34px rgba(2,6,23,.12);
}

.cardTopBlue{ border-top:5px solid var(--primary); }
.cardTopPurple{ border-top:5px solid var(--secondary); }
.cardTopGreen{ border-top:5px solid var(--success); }

.softGlowBlue:hover{ box-shadow:0 18px 34px rgba(2,6,23,.12), 0 0 20px rgba(37,99,235,.28); }
.softGlowPurple:hover{ box-shadow:0 18px 34px rgba(2,6,23,.12), 0 0 20px rgba(124,58,237,.28); }
.softGlowGreen:hover{ box-shadow:0 18px 34px rgba(2,6,23,.12), 0 0 20px rgba(22,163,74,.25); }

.smallMuted{ color:var(--muted); font-size:16px; margin-top:6px; }

.kBadge{
  display:inline-block;
  padding:6px 12px;
  border-radius:999px;
  background:rgba(37,99,235,.14);
  font-weight:600;
  margin-right:8px;
  margin-top:6px;
}

details summary{
  font-size:18px !important;
  font-weight:700;
}

hr{
  margin:12px 0 18px 0;
  opacity:.22;
}
</style>
""",
    unsafe_allow_html=True
)

# ======================================
# HERO (NO HTML LEAK)
# ======================================
components.html(
    """
<div style="
  padding:28px;
  border-radius:22px;
  background:linear-gradient(135deg, rgba(37,99,235,.22), rgba(124,58,237,.18), rgba(22,163,74,.14));
  border:1px solid rgba(15,23,42,.12);
  box-shadow:0 18px 44px rgba(2,6,23,.14);
  font-family: ui-sans-serif, system-ui;
">
  <div style="display:flex;justify-content:space-between;gap:18px;align-items:flex-start;flex-wrap:wrap;">
    <div>
      <h1 style="margin:0;color:#0F172A;">Multi-Dataset Classification Dashboard</h1>
      <div style="margin-top:8px;font-size:18px;opacity:.9;">
        Healthcare & Environment • Comparison • Prediction • Decision Support
      </div>
      <div style="margin-top:12px;">
        <span style="display:inline-block;padding:6px 12px;border-radius:999px;background:rgba(37,99,235,.16);font-weight:700;">Upload Data</span>
        <span style="display:inline-block;padding:6px 12px;border-radius:999px;background:rgba(124,58,237,.16);font-weight:700;margin-left:8px;">Choose Algorithm</span>
        <span style="display:inline-block;padding:6px 12px;border-radius:999px;background:rgba(22,163,74,.14);font-weight:700;margin-left:8px;">Interactive Insights</span>
      </div>
    </div>
  </div>
</div>
""",
    height=190
)

# ======================================
# SIDEBAR CONTROL (UPLOAD + DATASET + ALGO + MENU)
# ======================================
with st.sidebar:
    st.header("⚙️ Control Panel")

    dataset_mode = st.selectbox(
        "📌 Pilih Mode Dataset",
        ["Auto Detect", "Kesehatan (Breast Cancer)", "Lingkungan (ISPU Udara)"],
        index=0
    )

    uploaded = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

    st.markdown("---")
    menu = st.radio(
        "🧭 Navigation",
        ["About", "Steps", "Visualization", "Modeling", "Prediction", "Contact"],
        index=0
    )

    st.markdown("---")
    st.subheader("🤖 Pilih Algoritma (Prediction)")
    algo_choice = st.selectbox(
        "Algoritma yang digunakan saat prediksi",
        [
            "Logistic Regression",
            "KNN",
            "SVM",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting (opsional untuk lingkungan)"
        ],
        index=4
    )

# simpan agar semua modul bisa baca
st.session_state["dataset_mode"] = dataset_mode
st.session_state["uploaded_file"] = uploaded
st.session_state["algo_choice"] = algo_choice

# ======================================
# HIGHLIGHT CARDS (CLICKABLE EXPANDER)
# ======================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
<div class="card cardTopBlue softGlowBlue">
  <h3>🎯 Tujuan</h3>
  <div class="smallMuted">
    Klasifikasi 2 domain (kesehatan & lingkungan), komparasi algoritma,
    pilih model terbaik, prediksi & rekomendasi.
  </div>
  <details style="margin-top:10px;">
    <summary>Detail</summary>
    <div class="smallMuted" style="margin-top:10px;">
      Dashboard ini membangun <b>Decision Support System</b> berbasis Machine Learning:
      membantu interpretasi data dan memberi rekomendasi tindakan berdasarkan hasil prediksi.
    </div>
  </details>
</div>
""",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
<div class="card cardTopPurple softGlowPurple">
  <h3>🧠 Metodologi</h3>
  <div class="smallMuted">
    Preprocessing → EDA + Statistik → Modeling (komparasi) → Best Model → Prediction.
  </div>
  <details style="margin-top:10px;">
    <summary>Detail</summary>
    <div class="smallMuted" style="margin-top:10px;">
      Setiap dataset diproses secara kondisional (target berbeda), tetapi alur analitik sama.
      Model dapat dipilih manual, serta ditampilkan model terbaik berdasarkan metrik evaluasi.
    </div>
  </details>
</div>
""",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
<div class="card cardTopGreen softGlowGreen">
  <h3>📊 Output</h3>
  <div class="smallMuted">
    Visualisasi interaktif + interpretasi, tabel evaluasi model, aplikasi prediksi & rekomendasi.
  </div>
  <details style="margin-top:10px;">
    <summary>Detail</summary>
    <div class="smallMuted" style="margin-top:10px;">
      Metrik evaluasi: Accuracy, Precision, Recall, F1.
      Semua disajikan ringkas agar tidak perlu scroll panjang.
    </div>
  </details>
</div>
""",
        unsafe_allow_html=True
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ======================================
# RENDER MENU
# ======================================
if menu == "About":
    about.about_page()

elif menu == "Steps":
    steps.steps_page()

elif menu == "Visualization":
    visualisasi.visualization_page()

elif menu == "Modeling":
    modeling.modeling_page()

elif menu == "Prediction":
    prediction.prediction_page()

elif menu == "Contact":
    contact.contact_page()
