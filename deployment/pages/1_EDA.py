
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from wordcloud import WordCloud
import kagglehub
import os

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.markdown("---")

# ── Load dataset ───────────────────────────────────────────
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("andrewmvd/cyberbullying-classification")
    csv_file = os.path.join(path, 'cyberbullying_tweets.csv')
    return pd.read_csv(csv_file)

df = load_data()

# ── Sidebar filter ─────────────────────────────────────────
st.sidebar.header("⚙️ Filter")
all_categories = df['cyberbullying_type'].unique().tolist()
selected_categories = st.sidebar.multiselect(
    "Pilih kategori:",
    options=all_categories,
    default=all_categories
)

df_filtered = df[df['cyberbullying_type'].isin(selected_categories)]
st.sidebar.markdown(f"**Total tweet:** {len(df_filtered):,}")

# ══════════════════════════════════════════════════════════
# SECTION 1 — Distribusi Label
# ══════════════════════════════════════════════════════════

st.subheader("1️⃣ Distribusi Label Cyberbullying")

col1, col2 = st.columns(2)
label_counts = df_filtered['cyberbullying_type'].value_counts()
colors = ['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6','#1ABC9C']

with col1:
    chart_type = st.radio("Tipe chart:", ["Bar Chart", "Pie Chart"], horizontal=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    if chart_type == "Bar Chart":
        bars = ax.bar(label_counts.index, label_counts.values,
                      color=colors[:len(label_counts)], edgecolor='black')
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Jumlah Tweet")
        ax.set_title("Jumlah Tweet per Kategori")
        ax.tick_params(axis='x', rotation=30)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 20,
                    str(int(bar.get_height())),
                    ha='center', fontsize=9, fontweight='bold')
    else:
        ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%',
               colors=colors[:len(label_counts)], startangle=90)
        ax.set_title("Proporsi Kategori (%)")

    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("#### 📋 Ringkasan")
    summary = pd.DataFrame({
        "Kategori": label_counts.index,
        "Jumlah": label_counts.values,
        "Persentase (%)": (label_counts.values / label_counts.sum() * 100).round(2)
    }).reset_index(drop=True)
    st.dataframe(summary, use_container_width=True)

    st.markdown("#### 📌 Info Dataset")
    m1, m2 = st.columns(2)
    m1.metric("Total Tweet", f"{len(df_filtered):,}")
    m2.metric("Jumlah Kategori", len(label_counts))

st.markdown("---")

# ══════════════════════════════════════════════════════════
# SECTION 2 — Wordcloud per Kategori
# ══════════════════════════════════════════════════════════

st.subheader("2️⃣ WordCloud per Kategori")

selected_cat_wc = st.selectbox(
    "Pilih kategori untuk WordCloud:",
    options=selected_categories
)

subset_text = ' '.join(
    df_filtered[df_filtered['cyberbullying_type'] == selected_cat_wc]['tweet_text'].astype(str)
)

if subset_text.strip():
    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        colormap='Set2'
    ).generate(subset_text)

    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    ax_wc.set_title(f'WordCloud: {selected_cat_wc}', fontsize=14, fontweight='bold')
    st.pyplot(fig_wc)
    plt.close()
else:
    st.warning("Tidak ada teks untuk kategori ini.")

st.markdown("---")

# ══════════════════════════════════════════════════════════
# SECTION 3 — Top Kata per Kategori
# ══════════════════════════════════════════════════════════

st.subheader("3️⃣ Top Kata per Kategori")

col_left, col_right = st.columns([1, 2])

with col_left:
    selected_cat_top = st.selectbox(
        "Pilih kategori:",
        options=selected_categories,
        key="top_kata"
    )
    top_n = st.slider("Jumlah kata:", min_value=5, max_value=30, value=15)

with col_right:
    from sklearn.feature_extraction.text import CountVectorizer

    texts = df_filtered[df_filtered['cyberbullying_type'] == selected_cat_top]['tweet_text'].astype(str)
    cv = CountVectorizer(stop_words='english', max_features=top_n)

    try:
        cv_matrix = cv.fit_transform(texts)
        word_freq = pd.DataFrame({
            'Kata': cv.get_feature_names_out(),
            'Frekuensi': cv_matrix.toarray().sum(axis=0)
        }).sort_values('Frekuensi', ascending=True)

        fig_top, ax_top = plt.subplots(figsize=(7, top_n * 0.35 + 1))
        ax_top.barh(word_freq['Kata'], word_freq['Frekuensi'],
                    color='#3498DB', edgecolor='black')
        ax_top.set_title(f'Top {top_n} Kata — {selected_cat_top}', fontweight='bold')
        ax_top.set_xlabel('Frekuensi')
        st.pyplot(fig_top)
        plt.close()
    except Exception as e:
        st.warning(f"Tidak dapat menampilkan top kata: {e}")
