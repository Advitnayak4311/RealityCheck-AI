import streamlit as st
import numpy as np
import joblib
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# ------------------ CONFIG ------------------
st.set_page_config(page_title="RealityCheck AI", layout="centered")

# ------------------ LOAD MODELS ------------------
model_news = joblib.load("D:/RealityCheck-AI/models/fake_news_model.pkl")
model_url = joblib.load("D:/RealityCheck-AI/models/phishing_model.pkl")

with open("D:/RealityCheck-AI/models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# ------------------ FUNCTIONS ------------------

def extract_features(url):
    return [
        len(url),
        url.count('.'),
        int('@' in url),
        int('https' in url),
        int('-' in url),
        int('login' in url.lower())
    ]

def image_score(img):
    # Convert to grayscale
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    # Heuristic: AI images often smoother → low edges
    score = 1 - edge_density
    return min(max(score, 0), 1)

def calculate_risk(news_pred, url_pred, img_score):
    news_risk = 1 - news_pred
    url_risk = 1 - url_pred
    return (0.4 * news_risk) + (0.3 * url_risk) + (0.3 * img_score)

# ------------------ UI DESIGN ------------------

st.markdown("""
<style>
body {
    background-color: #0b0f19;
}
.stTextArea textarea, .stTextInput input {
    background-color: #1c2333;
    color: white;
    border-radius: 10px;
}
.stButton button {
    border-radius: 10px;
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🔍 RealityCheck AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Unified Trust & Risk Intelligence System</h4>", unsafe_allow_html=True)

# ------------------ INPUT ------------------

news_input = st.text_area("Enter News Text")
url_input = st.text_input("Enter URL")
image_file = st.file_uploader("Upload Image (Optional)", type=["jpg", "png", "jpeg"])

# ------------------ ANALYZE ------------------

if st.button("Analyze"):

    # News Prediction
    news_pred = model_news.predict(tfidf.transform([news_input]))[0]

    # URL Prediction
    url_pred = model_url.predict([extract_features(url_input)])[0]

    # Image Prediction
    if image_file:
        img = Image.open(image_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_pred = image_score(img)
    else:
        img_pred = 0.5  # neutral

    # Final Risk
    risk = calculate_risk(news_pred, url_pred, img_pred)

    # ------------------ RESULTS ------------------

    st.markdown("## 📊 Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("📰 News", "Real" if news_pred==1 else "Fake")
    col2.metric("🔗 URL", "Safe" if url_pred==1 else "Phishing")
    col3.metric("⚠️ Risk", round(risk,2))

    # Risk Bar
    st.progress(int(risk * 100))

    # Status
    if risk < 0.3:
        st.success("✅ SAFE")
    elif risk < 0.6:
        st.warning("⚠️ MEDIUM RISK")
    else:
        st.error("🚨 HIGH RISK")

    # ------------------ GRAPH ------------------

    st.markdown("### 📈 Risk Breakdown")

    labels = ['News Risk', 'URL Risk', 'Image Risk']
    values = [
        1 - news_pred,
        1 - url_pred,
        img_pred
    ]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim(0,1)
    st.pyplot(fig)