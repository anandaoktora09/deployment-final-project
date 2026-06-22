
import streamlit as st

st.set_page_config(
    page_title="Cyberbullying Classifier",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Cyberbullying Tweet Classifier")
st.markdown("**Final Project — Group 3 (Baby Python) | Data Science Batch 59 Digital Skola**")
st.markdown("---")

st.markdown("""
### Selamat Datang!

Aplikasi ini mengklasifikasikan tweet **berbahasa Inggris** apakah mengandung unsur **cyberbullying**
atau tidak, menggunakan metode **Hierarchical Classification** dengan TF-IDF + N-Gram.

#### 📌 Navigasi:
- 📊 **EDA** — Eksplorasi dan visualisasi dataset
- 🔍 **Classifier** — Deteksi cyberbullying pada tweet (Bahasa Inggris)

> Gunakan menu di **sidebar kiri** untuk berpindah halaman.
""")

st.info("💡 Tip: Klik halaman **Classifier** untuk langsung mencoba prediksi tweet!")
st.warning("⚠️ Model ini dioptimalkan untuk teks **Bahasa Inggris**. Input bahasa lain tetap akan diproses, namun akurasi tidak terjamin.")
