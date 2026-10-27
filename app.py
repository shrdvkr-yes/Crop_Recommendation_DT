import base64
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="centered")


# ---------- Load model & label encoder ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "crop_recommendation_model.pkl")
    encoder = joblib.load(BASE_DIR / "crop_label_encoder.pkl")
    return model, encoder


model, encoder = load_artifacts()


# ---------- Background image + CSS ----------
def find_background():
    """Finds your background photo (bg.jpg / bg.png / ...) in the app folder."""
    for name in ["bg.jpg", "bg.jpeg", "bg.png", "bg.webp",
                 "background.jpg", "background.jpeg", "background.png"]:
        p = BASE_DIR / name
        if p.exists():
            return p
    return None


def apply_background_and_style():
    css = (BASE_DIR / "style.css").read_text(encoding="utf-8")
    bg_rule = ""
    bg = find_background()
    if bg is not None:
        mime = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}[bg.suffix.lower()]
        encoded = base64.b64encode(bg.read_bytes()).decode()
        bg_rule = f"""
        .stApp {{
            background-image:
                linear-gradient(rgba(0, 0, 0, 0.45), rgba(0, 0, 0, 0.55)),
                url("data:image/{mime};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        """
    else:
        st.warning("Background photo not found. Save it as 'bg.jpg' in the same folder as app.py.")
    st.markdown(f"<style>{css}\n{bg_rule}</style>", unsafe_allow_html=True)


apply_background_and_style()

# ---------- Crop emojis ----------
EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘", "pigeonpeas": "🫛",
    "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🫘", "lentil": "🫘",
    "pomegranate": "🍎", "banana": "🍌", "mango": "🥭", "grapes": "🍇",
    "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎", "orange": "🍊",
    "papaya": "🥭", "coconut": "🥥", "cotton": "☁️", "jute": "🌿", "coffee": "☕",
}

# ---------- Header ----------
st.markdown("<h1 class='title'>🌱 Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Enter your soil and weather details and get the best crop to grow.</p>",
    unsafe_allow_html=True,
)

# ---------- Inputs ----------
with st.form("crop_form"):
    st.markdown("<h3 class='section'>🧪 Soil Nutrients</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    N = c1.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=90.0, step=1.0)
    P = c2.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=42.0, step=1.0)
    K = c3.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=43.0, step=1.0)

    st.markdown("<h3 class='section'>🌦️ Weather & Soil Conditions</h3>", unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    temperature = c4.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
    humidity = c5.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)

    c6, c7 = st.columns(2)
    ph = c6.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = c7.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=200.0, step=1.0)

    submitted = st.form_submit_button("🔍 Recommend Crop")

# ---------- Prediction ----------
if submitted:
    features = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )
    pred = model.predict(features)[0]
    crop = encoder.inverse_transform([pred])[0]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-emoji">{EMOJI.get(crop, "🌱")}</div>
            <div class="result-label">Recommended Crop</div>
            <div class="result-crop">{crop.title()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show top alternatives if the model supports probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        top = proba.argsort()[::-1][:3]
        top = [i for i in top if proba[i] > 0]
        if len(top) > 1:
            names = encoder.inverse_transform(model.classes_[top])
            items = "".join(
                f"<li>{EMOJI.get(n, '🌱')} {n.title()} — {proba[i] * 100:.0f}%</li>"
                for n, i in zip(names, top)
            )
            st.markdown(
                f"<div class='alt-card'><b>Top matches</b><ul>{items}</ul></div>",
                unsafe_allow_html=True,
            )

st.markdown("<p class='footer'>Built with Streamlit • Decision Tree Classifier</p>", unsafe_allow_html=True)