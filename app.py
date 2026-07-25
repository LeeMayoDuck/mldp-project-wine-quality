"""
Wine Quality Predictor - Streamlit app
Loads the tuned Random Forest model (wine_best_rf_model.pkl) trained in the
accompanying notebook and predicts whether a wine batch is likely to be rated
"good quality" (score >= 7) based on its physicochemical properties.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Page config & light styling
st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #FBF7F4; }
    h1, h2, h3 { color: #6B1E2E; }
    div.stButton > button {
        background-color: #6B1E2E;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.5em;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #8C2A3D;
        color: white;
    }
    .result-good, .result-good h3, .result-good p, .result-good b {
        color: #14361B !important;
    }
    .result-good {
        background-color: #EAF7EE;
        border-left: 6px solid #2E7D32;
        padding: 1.2em;
        border-radius: 8px;
    }
    .result-standard, .result-standard h3, .result-standard p, .result-standard b {
        color: #5C1712 !important;
    }
    .result-standard {
        background-color: #FDF0EF;
        border-left: 6px solid #B3261E;
        padding: 1.2em;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load model (cached so it only loads once per session)
MODEL_PATH = "wine_best_rf_model.pkl"


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# Header
st.title("Wine Quality Predictor")
st.markdown(
    """
    Estimate whether a wine batch is likely to be rated **"good quality"**
    (expert score ≥ 7) using lab-measured physicochemical properties -
    before it goes through a costly, time-consuming expert sensory panel.
    """
)

if not model_loaded:
    st.error(
        f"Could not find **{MODEL_PATH}**. Make sure this file is in the "
        "same folder as `app.py` (it's produced by running the training "
        "notebook, which saves the tuned model with `joblib.dump`)."
    )
    st.stop()

st.divider()


# Input form
st.subheader("Enter Batch Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    wine_type_label = st.selectbox("Wine Type", ["Red", "White"])
    fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 3.8, 16.0, 7.2, 0.1)
    volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.05, 1.60, 0.34, 0.01)
    citric_acid = st.slider("Citric Acid (g/dm³)", 0.0, 1.7, 0.32, 0.01)

with col2:
    residual_sugar = st.slider("Residual Sugar (g/dm³)", 0.5, 66.0, 5.4, 0.1)
    chlorides = st.slider("Chlorides (g/dm³)", 0.005, 0.62, 0.056, 0.001)
    free_so2 = st.slider("Free Sulfur Dioxide (mg/dm³)", 1.0, 290.0, 30.0, 1.0)
    total_so2 = st.slider("Total Sulfur Dioxide (mg/dm³)", 6.0, 440.0, 115.0, 1.0)

with col3:
    density = st.slider("Density (g/cm³)", 0.9870, 1.0390, 0.9947, 0.0001, format="%.4f")
    pH = st.slider("pH", 2.7, 4.1, 3.22, 0.01)
    sulphates = st.slider("Sulphates (g/dm³)", 0.2, 2.0, 0.53, 0.01)
    alcohol = st.slider("Alcohol (% by volume)", 8.0, 15.0, 10.5, 0.1)


# Input validation
errors = []
if free_so2 > total_so2:
    errors.append("Free sulfur dioxide cannot exceed total sulfur dioxide.")
if any(
    v <= 0
    for v in [fixed_acidity, residual_sugar, chlorides, free_so2, total_so2, density, sulphates, alcohol]
):
    errors.append("All measurements must be greater than zero.")

if errors:
    for e in errors:
        st.warning(f"⚠️ {e}")


# Build the feature row
wine_type_val = 0 if wine_type_label == "Red" else 1
total_acidity = fixed_acidity + volatile_acidity
so2_ratio = free_so2 / total_so2 if total_so2 > 0 else 0.0

input_df = pd.DataFrame(
    [
        {
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
            "wine_type": wine_type_val,
            "total_acidity": total_acidity,
            "so2_ratio": so2_ratio,
        }
    ]
)

st.divider()


# Predict
predict_clicked = st.button("Predict Quality", disabled=bool(errors))

if predict_clicked:
    try:
        # Align column order to what the model expects, if available
        if hasattr(model, "feature_names_in_"):
            input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Result")

        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-good">
                <h3>✅ Likely Good Quality</h3>
                <p>Model confidence: <b>{probability:.1%}</b> probability of
                being rated "good quality" (score ≥ 7).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-standard">
                <h3>Standard Quality</h3>
                <p>Model confidence: <b>{probability:.1%}</b> probability of
                being rated "good quality" - below the threshold to flag this
                batch as high-potential.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(float(probability))

        with st.expander("See input summary"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(
            "❌ Something went wrong while generating the prediction. "
            f"Details: {e}"
        )

st.divider()
st.caption(
    "Model: Random Forest, tuned via RandomizedSearchCV (F1-optimized). "
    "For decision support only - not a substitute for expert sensory evaluation."
)