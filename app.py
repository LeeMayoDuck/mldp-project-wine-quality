"""
Wine Quality Predictor - Streamlit app
Loads the tuned Random Forest model (wine_best_rf_model.pkl) and predicts,
live as you move the sliders, whether a wine batch is likely to be rated
"good quality" (score >= 7) based on its physicochemical properties.
"""

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="\U0001F377",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design tokens, styling & animations
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

    [data-testid="stAppViewContainer"], .stApp { background-color: var(--paper) !important; }
    section.main > div { padding-top: 1.5rem; }

    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    .hero-title {
        font-family: 'Fraunces', serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.01em;
    }

    /* Color is forced on all text-bearing elements, but font-family is kept
       separate and excludes Streamlit's icon elements (data-testid
       "stIconMaterial", or any class hinting at material icons/symbols) -
       those render as ligature text using a special icon font, so forcing
       them onto Inter turns the icon glyph into visible literal text (e.g.
       the expander arrow showing up as the word "arrow_right" overlapping
       its label). */
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] div,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: var(--ink) !important;
    }
    [data-testid="stAppViewContainer"] p:not([data-testid="stIconMaterial"]):not([class*="material"]),
    [data-testid="stAppViewContainer"] div:not([data-testid="stIconMaterial"]):not([class*="material"]),
    [data-testid="stAppViewContainer"] span:not([data-testid="stIconMaterial"]):not([class*="material"]),
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-family: 'Inter', sans-serif;
    }

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

    /* Streamlit's hamburger menu renders in a BaseWeb popover portal, always
       on a dark background regardless of the app's own light theme, so its
       text needs to be forced white. The one in-app element that also uses
       a BaseWeb popover is the slider's drag tooltip (the number that pops
       up above the thumb) - that one sits on our light card background, so
       it's excluded here and left to the --ink rule further up. */
    div[data-baseweb="popover"] *:not([data-testid="stThumbValue"]),
    [data-testid="stMainMenuPopover"],
    [data-testid="stMainMenuPopover"] *,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] *,
    ul[role="menu"],
    ul[role="menu"] *,
    li[role="menuitem"],
    li[role="menuitem"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* The toolbar (Deploy button, hamburger icon) sits in Streamlit's own
       dark header chrome, which stays dark regardless of the app's theme -
       var(--text-color) wasn't resolving to white there, so it's hardcoded. */
    [data-testid="stAppDeployButton"] button,
    [data-testid="stAppDeployButton"] span,
    [data-testid="stMainMenuButton"],
    [data-testid="stMainMenuButton"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    /* ---------------------------------------------------------------------
       Animations
       --------------------------------------------------------------------- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        0%   { opacity: 0; transform: scale(0.94) translateY(6px); }
        60%  { opacity: 1; transform: scale(1.015) translateY(0); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-10px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes softPulseGood {
        0%, 100% { box-shadow: 0 0 0 0 rgba(47, 92, 54, 0.0); }
        50%      { box-shadow: 0 0 0 8px rgba(47, 92, 54, 0.08); }
    }
    @keyframes ringPop {
        from { transform: scale(0.85); opacity: 0; }
        to   { transform: scale(1); opacity: 1; }
    }

    .hero-title, .hero-sub, .eyebrow {
        animation: fadeInUp 0.55s ease-out both;
    }
    .hero-sub { animation-delay: 0.08s; }

    button[data-baseweb="tab"] p,
    div[data-baseweb="tab-highlight"],
    div[data-testid="stExpander"] summary,
    .stSlider [role="slider"] {
        transition: color 0.25s ease, background-color 0.25s ease,
                    transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stSlider [role="slider"]:hover,
    .stSlider [role="slider"]:focus {
        transform: scale(1.15);
        box-shadow: 0 0 0 6px rgba(110, 30, 51, 0.12);
    }

    button[data-baseweb="tab"] {
        transition: transform 0.2s ease;
    }
    button[data-baseweb="tab"]:hover p {
        transform: translateY(-1px);
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
        animation: popIn 0.4s cubic-bezier(0.22, 1, 0.36, 1) both;
        transition: box-shadow 0.3s ease;
    }
    .scorecard.is-good {
        animation: popIn 0.4s cubic-bezier(0.22, 1, 0.36, 1) both,
                   softPulseGood 2.2s ease-in-out 0.4s infinite;
    }

    .ring-wrap {
        animation: ringPop 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
        animation-delay: 0.05s;
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
        animation: slideInLeft 0.3s ease-out both;
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

    div[data-testid="stDataFrame"] {
        background-color: var(--card) !important;
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
        }
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
    with st.spinner("Loading model..."):
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
# Live scorecard - recomputes on every rerun (i.e. every slider move).
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

            verdict_class = "verdict-good" if prediction == 1 else "verdict-standard"
            verdict_text = "Likely Good Quality" if prediction == 1 else "Standard Quality"
            caption_text = (
                "Strong candidate for premium bottling or further review."
                if prediction == 1
                else "Below the threshold to flag as high-potential."
            )
            card_class = "scorecard is-good" if prediction == 1 else "scorecard"

            prev_probability = st.session_state.get("_prev_probability", 0.0)

            # Rendered via components.html (real iframe) rather than
            # st.markdown, so the <script> below actually executes and can
            # animate the ring. The iframe is its own document with no
            # access to the app's CSS variables, so colors are repeated here
            # as literal values.
            scorecard_html = f"""
            <html>
            <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');
                html, body {{
                    margin: 0; padding: 0; background: transparent;
                    font-family: 'Inter', sans-serif;
                    overflow: hidden;
                }}
                .scorecard {{
                    background: #FFFFFF;
                    border: 1px solid #E4D9CE;
                    border-radius: 14px;
                    padding: 1.6rem 1.8rem;
                    box-sizing: border-box;
                    animation: popIn 0.4s cubic-bezier(0.22, 1, 0.36, 1) both;
                }}
                .scorecard.is-good {{
                    animation: popIn 0.4s cubic-bezier(0.22, 1, 0.36, 1) both,
                               softPulseGood 2.2s ease-in-out 0.4s infinite;
                }}
                @keyframes popIn {{
                    0%   {{ opacity: 0; transform: scale(0.94) translateY(6px); }}
                    60%  {{ opacity: 1; transform: scale(1.015) translateY(0); }}
                    100% {{ opacity: 1; transform: scale(1) translateY(0); }}
                }}
                @keyframes softPulseGood {{
                    0%, 100% {{ box-shadow: 0 0 0 0 rgba(47, 92, 54, 0.0); }}
                    50%      {{ box-shadow: 0 0 0 8px rgba(47, 92, 54, 0.08); }}
                }}
                .eyebrow {{
                    font-family: 'IBM Plex Mono', monospace;
                    font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
                    color: #8A6A3D; font-weight: 600; margin-bottom: 0.3rem;
                }}
                .row {{ display: flex; align-items: center; gap: 1.4rem; }}
                .verdict-title {{
                    font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 600;
                    margin: 0.2rem 0 0.1rem 0;
                }}
                .verdict-good {{ color: #2F5C36; }}
                .verdict-standard {{ color: #8A2E24; }}
                .verdict-caption {{ color: #5A4A50; font-size: 0.92rem; margin-bottom: 0; }}
                @media (prefers-reduced-motion: reduce) {{
                    *, *::before, *::after {{
                        animation-duration: 0.001ms !important;
                        animation-iteration-count: 1 !important;
                    }}
                }}
            </style>
            </head>
            <body>
            <div class="{card_class}">
              <div class="eyebrow">Live &middot; updates as you adjust sliders</div>
              <div class="row">
                <div>
                  <svg width="140" height="140" viewBox="0 0 140 140" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="70" cy="70" r="{radius}" fill="none" stroke="#E4D9CE" stroke-width="12"/>
                    <circle id="ring-progress" cx="70" cy="70" r="{radius}" fill="none" stroke="{ring_color}"
                      stroke-width="12" stroke-linecap="round"
                      stroke-dasharray="{circumference:.2f} {circumference:.2f}"
                      stroke-dashoffset="{circumference:.2f}"
                      transform="rotate(-90 70 70)"/>
                    <text id="ring-pct" x="70" y="65" text-anchor="middle" font-family="IBM Plex Mono, monospace"
                      font-size="24" font-weight="600" fill="#2B1420">0%</text>
                    <text x="70" y="84" text-anchor="middle" font-family="Inter, sans-serif"
                      font-size="11" fill="#5A4A50">confidence</text>
                  </svg>
                </div>
                <div>
                  <div class="verdict-title {verdict_class}">{verdict_text}</div>
                  <div class="verdict-caption">{caption_text}</div>
                </div>
              </div>
            </div>
            <script>
              var targetPct = {probability:.6f};
              var startPct = {prev_probability:.6f};
              var circumference = {circumference:.2f};
              var circle = document.getElementById('ring-progress');
              var label = document.getElementById('ring-pct');

              function paint(p) {{
                circle.setAttribute('stroke-dashoffset', (circumference * (1 - p)).toFixed(2));
                label.textContent = Math.round(p * 100) + '%';
              }}

              var reduceMotion = window.matchMedia &&
                window.matchMedia('(prefers-reduced-motion: reduce)').matches;

              if (reduceMotion) {{
                paint(targetPct);
              }} else {{
                var duration = 700;
                var startTime = null;
                function ease(t) {{ return 1 - Math.pow(1 - t, 3); }}
                function frame(ts) {{
                  if (!startTime) startTime = ts;
                  var t = Math.min((ts - startTime) / duration, 1);
                  paint(startPct + (targetPct - startPct) * ease(t));
                  if (t < 1) requestAnimationFrame(frame);
                }}
                requestAnimationFrame(frame);
              }}
            </script>
            </body>
            </html>
            """

            components.html(scorecard_html, height=210, scrolling=False)

            # One-off celebration only on the transition into "good quality",
            # not on every rerun.
            if prediction == 1 and st.session_state.get("_prev_prediction") != 1:
                st.balloons()
            st.session_state["_prev_prediction"] = int(prediction)
            st.session_state["_prev_probability"] = float(probability)

            with st.expander("See input summary"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Something went wrong while generating the prediction. Details: {e}")

st.markdown('<hr class="hairline">', unsafe_allow_html=True)