"""
Wine Quality Predictor - Streamlit app
Loads the tuned Random Forest model (wine_best_rf_model.pkl) and predicts,
live as you move the sliders, whether a wine batch is likely to be rated
"good quality" (score >= 7) based on its physicochemical properties.
"""

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="\U0001F377",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design tokens & styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --ink: #2B1420;
        --ink-soft: #5A4A50;
        --paper: #FAF4EE;
        --card: #FFFFFF;
        --bordeaux: #6E1E33;
        --bordeaux-deep: #4A1424;
        --gold: #8A6A3D;
        --success: #2F5C36;
        --alert: #8A2E24;
        --hairline: #E4D9CE;
    }

    /* Force the whole app - including Streamlit's own widget wrappers - onto
       the light palette. Streamlit's default dark-theme text color otherwise
       leaks through on slider/tab labels regardless of generic tag selectors,
       so every relevant data-testid is targeted explicitly with !important. */
    [data-testid="stAppViewContainer"], .stApp { background-color: var(--paper) !important; }
    section.main > div { padding-top: 1.5rem; }

    h1, h2, h3, .hero-title,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-family: 'Fraunces', serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.01em;
    }

    p, div, span,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-family: 'Inter', sans-serif;
        color: var(--ink) !important;
    }

    /* Slider labels + numeric readout - the specific elements that were washed out */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stSlider"] label,
    [data-testid="stSliderTickBarMin"],
    [data-testid="stSliderTickBarMax"],
    [data-testid="stThumbValue"] {
        color: var(--ink) !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }

    .hero-title { font-size: 2.6rem; font-weight: 600; margin-bottom: 0.1rem; }
    .hero-sub { color: var(--ink-soft) !important; font-size: 1.02rem; max-width: 640px; line-height: 1.5; }
    .hairline { border: none; border-top: 1px solid var(--hairline); margin: 1.6rem 0; }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--gold) !important;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    button[data-baseweb="tab"] p { color: var(--ink-soft) !important; font-weight: 500 !important; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: var(--bordeaux) !important; font-weight: 600 !important; }
    div[data-baseweb="tab-highlight"] { background-color: var(--bordeaux) !important; }

    .scorecard {
        background: var(--card);
        border: 1px solid var(--hairline);
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
    }

    .verdict-good { color: var(--success) !important; }
    .verdict-standard { color: var(--alert) !important; }
    .verdict-title { font-family: 'Fraunces', serif !important; font-size: 1.5rem; font-weight: 600; margin: 0.2rem 0 0.1rem 0; }
    .verdict-caption { color: var(--ink-soft) !important; font-size: 0.92rem; margin-bottom: 0.6rem; }
    .confidence-number { font-family: 'IBM Plex Mono', monospace !important; font-weight: 600; font-size: 1.05rem; color: var(--ink) !important; }

    .warning-banner {
        background: #FBF3E3;
        border-left: 4px solid var(--gold);
        padding: 0.7rem 1rem;
        border-radius: 6px;
        color: var(--ink) !important;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
    }

    .stCaptionContainer, [data-testid="stCaptionContainer"] p { color: var(--ink-soft) !important; }
    div[data-testid="stExpander"] {
    background-color: var(--card) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 10px !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: var(--card) !important;
        color: var(--ink) !important;
    }
    div[data-testid="stExpander"] summary:hover,
    div[data-testid="stExpander"] summary:focus,
    div[data-testid="stExpander"] details[open] summary {
        background-color: var(--paper) !important;
        color: var(--ink) !important;
    }
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        background-color: var(--card) !important;
    }

    /* st.dataframe inside the expander was hitting the same dark-background issue */
    div[data-testid="stDataFrame"] {
        background-color: var(--card) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = "wine_best_rf_model.pkl"


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Decision support &middot; Random Forest</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Wine Quality Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Estimate whether a batch is likely to be rated '
    '<b>&ldquo;good quality&rdquo;</b> (expert score &ge; 7) from lab-measured '
    'physicochemical properties &ndash; before it goes through a costly, '
    'time-consuming expert sensory panel. The scorecard updates live as you '
    'move the sliders.</div>',
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(
        f"Could not find **{MODEL_PATH}**. Make sure this file sits in the same "
        "folder as `app.py` &ndash; it's produced by running the training "
        "notebook, which saves the tuned model with `joblib.dump`."
    )
    st.stop()

st.markdown('<hr class="hairline">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Layout: inputs on the left, live scorecard on the right
# ---------------------------------------------------------------------------
left, right = st.columns([1.3, 1])

with left:
    st.markdown("### Batch Measurements")

    tab1, tab2, tab3 = st.tabs(["Acidity & Balance", "Sulfur & Preservation", "Body & Alcohol"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fixed_acidity = st.slider("Fixed acidity (g/dm\u00b3)", 3.8, 16.0, 7.2, 0.1)
            volatile_acidity = st.slider("Volatile acidity (g/dm\u00b3)", 0.05, 1.60, 0.34, 0.01)
        with c2:
            citric_acid = st.slider("Citric acid (g/dm\u00b3)", 0.0, 1.7, 0.32, 0.01)
            pH = st.slider("pH", 2.7, 4.1, 3.22, 0.01)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            free_so2 = st.slider("Free sulfur dioxide (mg/dm\u00b3)", 1.0, 290.0, 30.0, 1.0)
            sulphates = st.slider("Sulphates (g/dm\u00b3)", 0.2, 2.0, 0.53, 0.01)
        with c2:
            total_so2 = st.slider("Total sulfur dioxide (mg/dm\u00b3)", 6.0, 440.0, 115.0, 1.0)
            chlorides = st.slider("Chlorides (g/dm\u00b3)", 0.005, 0.62, 0.056, 0.001)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            residual_sugar = st.slider("Residual sugar (g/dm\u00b3)", 0.5, 66.0, 5.4, 0.1)
            density = st.slider("Density (g/cm\u00b3)", 0.9870, 1.0390, 0.9947, 0.0001, format="%.4f")
        with c2:
            alcohol = st.slider("Alcohol (% by volume)", 8.0, 15.0, 10.5, 0.1)

    errors = []
    if free_so2 > total_so2:
        errors.append("Free sulfur dioxide cannot exceed total sulfur dioxide.")
    if any(v <= 0 for v in [fixed_acidity, residual_sugar, chlorides, free_so2, total_so2, density, sulphates, alcohol]):
        errors.append("All measurements must be greater than zero.")

    for e in errors:
        st.markdown(f'<div class="warning-banner">&#9888; {e}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Build feature row (no wine_type - dropped during feature engineering)
# ---------------------------------------------------------------------------
total_acidity = fixed_acidity + volatile_acidity
so2_ratio = free_so2 / total_so2 if total_so2 > 0 else 0.0

input_df = pd.DataFrame(
    [{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_so2,
        "total sulfur dioxide": total_so2,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "total_acidity": total_acidity,
        "so2_ratio": so2_ratio,
    }]
)

# ---------------------------------------------------------------------------
# Live scorecard - recomputes on every rerun (i.e. every slider move),
# no button needed since Streamlit already reruns the script automatically
# ---------------------------------------------------------------------------
with right:
    st.markdown("### Live Scorecard")

    if errors:
        st.markdown(
            '<div class="scorecard" style="color: var(--ink-soft);">'
            'Fix the warning(s) on the left to see a live prediction.</div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            model_input = input_df
            if hasattr(model, "feature_names_in_"):
                model_input = model_input[model.feature_names_in_]

            prediction = model.predict(model_input)[0]
            probability = model.predict_proba(model_input)[0][1]
            pct = probability * 100

            ring_color = "#2F5C36" if prediction == 1 else "#8A2E24"
            radius = 52
            circumference = 2 * 3.14159265 * radius
            dash = circumference * probability
            gap = circumference - dash

            ring_svg = f"""
            <svg width="140" height="140" viewBox="0 0 140 140" xmlns="http://www.w3.org/2000/svg">
              <circle cx="70" cy="70" r="{radius}" fill="none" stroke="#E4D9CE" stroke-width="12"/>
              <circle cx="70" cy="70" r="{radius}" fill="none" stroke="{ring_color}"
                stroke-width="12" stroke-linecap="round"
                stroke-dasharray="{dash:.1f} {gap:.1f}"
                transform="rotate(-90 70 70)"/>
              <text x="70" y="65" text-anchor="middle" font-family="IBM Plex Mono, monospace"
                font-size="24" font-weight="600" fill="#2B1420">{pct:.0f}%</text>
              <text x="70" y="84" text-anchor="middle" font-family="Inter, sans-serif"
                font-size="11" fill="#5A4A50">confidence</text>
            </svg>
            """

            verdict_class = "verdict-good" if prediction == 1 else "verdict-standard"
            verdict_text = "Likely Good Quality" if prediction == 1 else "Standard Quality"
            caption_text = (
                "Strong candidate for premium bottling or further review."
                if prediction == 1
                else "Below the threshold to flag as high-potential."
            )

            st.markdown(
                f"""
                <div class="scorecard">
                  <div class="eyebrow">Live &middot; updates as you adjust sliders</div>
                  <div style="display:flex; align-items:center; gap:1.4rem;">
                    <div>{ring_svg}</div>
                    <div>
                      <div class="verdict-title {verdict_class}">{verdict_text}</div>
                      <div class="verdict-caption">{caption_text}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("See input summary"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Something went wrong while generating the prediction. Details: {e}")

st.markdown('<hr class="hairline">', unsafe_allow_html=True)
st.caption(
    "Model: Random Forest, tuned via RandomizedSearchCV (F1-optimized). "
    "For decision support only &ndash; not a substitute for expert sensory evaluation."
)