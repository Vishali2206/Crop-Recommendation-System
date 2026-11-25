import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Page configuration and model loading
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgriSmart | Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = Path(__file__).with_name("crop_recommendation_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.exception(exc)
    st.stop()

# These are the 22 classes actually present in the supplied trained model.
crop_classes = [str(c) for c in model.classes_]

# ---------------------------------------------------------
# Dashboard styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --green: #2f9e62;
        --green-dark: #176b45;
        --mint: #dff5e8;
        --ink: #173629;
        --muted: #61776b;
        --panel: #ffffff;
        --line: #e1eee6;
    }

    .stApp {
        background:
          radial-gradient(circle at 8% 0%, rgba(173, 226, 190, .32), transparent 27%),
          linear-gradient(180deg, #f5fbf6 0%, #edf6ef 100%);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(245,251,246,.86); }
    .block-container { padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1240px; }

    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 { color: var(--ink); }

    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 26px;
        background: linear-gradient(115deg, #123f2c 0%, #176b45 58%, #2f9e62 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(23, 85, 55, .18);
        position: relative;
        overflow: hidden;
        margin-bottom: 1.35rem;
    }
    .hero:after {
        content: "🌿";
        position: absolute;
        right: 4%;
        top: -22px;
        font-size: 9rem;
        opacity: .13;
        transform: rotate(-18deg);
    }
    .hero-kicker {
        color: #bdebd0;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .hero h1 {
        color: #fff !important;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.1rem, 5vw, 3.2rem);
        font-weight: 800;
        line-height: 1.08;
        margin: 0;
    }
    .hero p {
        color: #e2f5e9 !important;
        font-size: 1.03rem;
        margin: .7rem 0 0 0;
        max-width: 690px;
    }
    .hero-chip {
        display: inline-block;
        margin-top: 1.1rem;
        padding: .42rem .8rem;
        border: 1px solid rgba(255,255,255,.27);
        border-radius: 999px;
        color: #effff4;
        font-size: .78rem;
        background: rgba(255,255,255,.10);
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: #173629;
        font-weight: 800;
        font-size: 1.35rem;
        margin: .35rem 0 .2rem 0;
    }
    .section-sub {
        color: #688074;
        margin-bottom: 1rem;
        font-size: .93rem;
    }
    .panel {
        background: rgba(255,255,255,.92);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 8px 25px rgba(32, 79, 52, .055);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 1rem 1.15rem;
        min-height: 98px;
        box-shadow: 0 7px 20px rgba(32,79,52,.045);
    }
    .metric-label { color: #6b8276; font-size: .78rem; font-weight: 700; }
    .metric-value { color: #176b45; font-size: 1.55rem; font-weight: 800; margin-top: .2rem; }
    .result-card {
        background: linear-gradient(130deg, #e5f8eb, #f7fff9);
        border: 1px solid #bfe8cc;
        border-radius: 22px;
        padding: 1.5rem;
        margin: .7rem 0 1rem 0;
    }
    .result-label {
        color: #397451;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .result-crop {
        color: #145b39;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 5vw, 2.8rem);
        font-weight: 800;
        margin: .2rem 0;
        text-transform: capitalize;
    }
    .result-note { color: #557565; font-size: .9rem; }
    .stButton > button {
        background: linear-gradient(100deg, #197348, #2f9e62);
        color: white;
        border: 0;
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 800;
        box-shadow: 0 8px 18px rgba(25,115,72,.17);
        transition: .2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(100deg, #125b38, #258451);
        color: white;
        border: 0;
        transform: translateY(-1px);
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #d5e7db;
        background: #fbfefc;
    }
    label { color: #294d3a !important; font-weight: 700 !important; }
    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 15px;
        background: rgba(255,255,255,.8);
    }
    .crop-pill {
        display: inline-block;
        background: #e8f6ec;
        color: #246644;
        border: 1px solid #ccebd5;
        padding: .38rem .68rem;
        border-radius: 999px;
        margin: .22rem .18rem;
        font-size: .82rem;
        font-weight: 700;
    }
    .footer {
        color: #71877b;
        text-align: center;
        font-size: .82rem;
        padding: 1.4rem 0 .4rem 0;
    }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">Smart agriculture • Machine learning</div>
  <h1>AgriSmart 🌱</h1>
  <p>Find a suitable crop using soil nutrients and local environmental
  conditions. Enter the seven parameters to get a model-based recommendation.</p>
  <span class="hero-chip">🌾 Crop Recommendation Dashboard</span>
</div>
""", unsafe_allow_html=True)

# Dashboard summary
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">SUPPORTED CROP CLASSES</div><div class="metric-value">{len(crop_classes)}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-label">INPUT PARAMETERS</div><div class="metric-value">07</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-label">MODEL TYPE</div><div class="metric-value">Random Forest</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.12, .88], gap="large")

with left:
    st.markdown('<div class="section-heading">🌿 Soil & Climate Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter the measured values. Use the same units as the dataset.</div>', unsafe_allow_html=True)

    with st.form("crop_input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            N = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0, step=1.0, help="Soil nitrogen value")
            P = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0, step=1.0, help="Soil phosphorus value")
            K = st.number_input("Potassium (K)", min_value=0.0, value=43.0, step=1.0, help="Soil potassium value")
        with c2:
            temperature = st.number_input("Temperature (°C)", value=20.87, step=0.5)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
        with c3:
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.50, step=0.1)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=202.93, step=1.0)

        submitted = st.form_submit_button("🌾  Find Suitable Crop", use_container_width=True)

with right:
    st.markdown('<div class="section-heading">📋 How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
      <b style="color:#176b45">01 · Add values</b><br>
      <span style="color:#61776b">Provide N, P, K, temperature, humidity, pH and rainfall.</span><br><br>
      <b style="color:#176b45">02 · Run prediction</b><br>
      <span style="color:#61776b">The saved Random Forest model compares your input with patterns learned during training.</span><br><br>
      <b style="color:#176b45">03 · Review results</b><br>
      <span style="color:#61776b">See the top model predictions and their relative scores.</span>
    </div>
    """, unsafe_allow_html=True)
    st.info("For meaningful results, enter values measured for the field and use consistent units.")

# ---------------------------------------------------------
# Prediction output
# ---------------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        top_indices = probabilities.argsort()[::-1][:5]

        st.markdown("---")
        st.markdown('<div class="section-heading">✨ Your Crop Recommendation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Top recommended crop by the trained model</div>
          <div class="result-crop">🌾 {str(prediction).replace('_', ' ')}</div>
          <div class="result-note">Prediction generated from the seven values you entered.</div>
        </div>
        """, unsafe_allow_html=True)

        result_col, score_col = st.columns([.9, 1.1], gap="large")
        with result_col:
            st.markdown("**Input summary**")
            summary = pd.DataFrame({
                "Parameter": ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "Soil pH", "Rainfall"],
                "Value": [N, P, K, temperature, humidity, ph, rainfall],
                "Unit": ["", "", "", "°C", "%", "", "mm"]
            })
            st.dataframe(summary, hide_index=True, use_container_width=True)
        with score_col:
            st.markdown("**Top 5 model predictions**")
            for rank, idx in enumerate(top_indices, start=1):
                crop = str(classes[idx]).replace("_", " ").title()
                score = float(probabilities[idx]) * 100
                st.markdown(f"**{rank}. {crop}** · {score:.1f}%")
                st.progress(min(max(score / 100, 0.0), 1.0))

        st.caption("These percentages are the model's class scores, not guarantees of crop yield, profitability, or farming success. Confirm recommendations with local agricultural experts and field conditions.")

    except Exception as exc:
        st.error("Prediction could not be completed. Please check the entered values and model file.")
        st.exception(exc)

# ---------------------------------------------------------
# Complete class coverage shown from the trained model
# ---------------------------------------------------------
with st.expander(f"🌾 View all {len(crop_classes)} crops supported by this model"):
    st.write("The dashboard can predict the following crop labels because these are the classes stored in the supplied trained model:")
    pills = "".join(
        f'<span class="crop-pill">{str(c).replace("_", " ").title()}</span>'
        for c in crop_classes
    )
    st.markdown(pills, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  AgriSmart · Machine Learning Crop Recommendation System<br>
  A decision-support tool — not a substitute for local agronomy advice.
</div>
""", unsafe_allow_html=True)
