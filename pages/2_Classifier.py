
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from contractions import fix
from langdetect import detect, LangDetectException

nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

st.set_page_config(page_title="Classifier", page_icon="🔍", layout="centered")

# ── Load 4 file model ──────────────────────────────────────
@st.cache_resource
def load_models():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model_stage1_en.pkl", "rb") as f:
        model_stage1_en = pickle.load(f)
    with open("model_stage2_en.pkl", "rb") as f:
        model_stage2_en = pickle.load(f)
    with open("encoder_stage2_en.pkl", "rb") as f:
        encoder_stage2_en = pickle.load(f)
    return vectorizer, model_stage1_en, model_stage2_en, encoder_stage2_en

vectorizer, model_stage1_en, model_stage2_en, encoder_stage2_en = load_models()

# ── Setup preprocessing ────────────────────────────────────
lemmatizer_en = WordNetLemmatizer()
stop_wordsENCust = set(stopwords.words('english'))

def clean_text(text):
    if text is None or (isinstance(text, float)):
        return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def detect_language_safe(text):
    s = str(text).strip()
    if not s:
        return None
    word_count = len(s.split())
    if word_count < 4:
        return None
    try:
        return detect(s)
    except LangDetectException:
        return None
    except Exception:
        return None

def preprocess_text(text):
    cleaned = clean_text(text)
    text_fixed = fix(cleaned)
    words = cleaned.split()
    tokens = [lemmatizer_en.lemmatize(w) for w in words if w not in stop_wordsENCust]
    return tokens

def predict_hierarchical(text):
    lang = detect_language_safe(text)
    is_non_english = lang not in ['en', None]

    processed_tokens = preprocess_text(text)
    processed_text = ' '.join(processed_tokens)
    vec = vectorizer.transform([processed_text])

    pred_s1 = model_stage1_en.predict(vec)[0]
    if pred_s1 == 0:
        return "not_cyberbullying", None, is_non_english

    pred_s2 = model_stage2_en.predict(vec)[0]
    label = encoder_stage2_en.inverse_transform([pred_s2])[0]
    return "cyberbullying", label, is_non_english

# ── UI ─────────────────────────────────────────────────────
st.title("🔍 Cyberbullying Tweet Classifier")
st.markdown("**Final Project — Group 3 (Baby Python) | Data Science Batch 59 Digital Skola**")
st.markdown("---")

st.markdown("### Masukkan teks tweet di bawah ini (Bahasa Inggris):")
user_input = st.text_area("", placeholder="Type your tweet here...", height=150)

color_map = {
    "age":                "🟠",
    "ethnicity":          "🟣",
    "gender":             "🔵",
    "religion":           "🟡",
    "other_cyberbullying":"🔴"
}

if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong!")
    else:
        with st.spinner("Menganalisis tweet..."):
            result, category, is_non_english = predict_hierarchical(user_input)

        st.markdown("---")

        if is_non_english:
            st.warning("⚠️ Tweet terdeteksi **bukan Bahasa Inggris**. Model ini hanya mendukung Bahasa Inggris, sehingga hasil prediksi tidak ditampilkan. Silakan masukkan tweet berbahasa Inggris.")
        else:
            st.markdown("### Hasil Prediksi:")

            if result == "not_cyberbullying":
                st.success("🟢 **Not Cyberbullying**")
                st.info("✅ Tweet ini tidak mengandung unsur cyberbullying.")
            else:
                emoji = color_map.get(category, "⚪")
                st.error("⚠️ Tweet ini terdeteksi mengandung cyberbullying!")
                st.markdown(f"**Kategori:** {emoji} **{category.replace('_', ' ').title()}**")
