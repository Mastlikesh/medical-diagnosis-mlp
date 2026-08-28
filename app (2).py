"""
Streamlit Human-in-the-Loop Verification Dashboard
Loads the trained MLP model and lets a user enter patient values to get a
prediction. Low-confidence predictions are automatically flagged for manual
clinical review — this is the exact gap your literature survey identifies
("Absence of Real-Time Validation Interfaces" / "Isolation of Black-Box AI").

RUN LOCALLY:
    pip install -r requirements.txt
    streamlit run app.py

DEPLOY (free): push this folder to a public GitHub repo, then go to
share.streamlit.io -> New app -> pick the repo -> set main file to app.py.
"""

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go

st.set_page_config(page_title="Intelligent Medical Diagnosis Dashboard", layout="wide")

# ── Load model + preprocessing artifacts ─────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("mlp_diagnosis_model.keras")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
except Exception as e:
    st.error(
        "Model files not found. Run train_model.py first (in Colab), then copy "
        "mlp_diagnosis_model.keras, scaler.pkl, and feature_names.pkl into this folder."
    )
    st.stop()

CONFIDENCE_THRESHOLD = 0.75  # below this, route to human review

# ── Radar chart: visualize the patient's Mean / SE / Worst measurements ──
def get_radar_chart(inputs, feature_names, scaler):
    """Groups the 30 features into Mean / Standard Error / Worst (10 each,
    same order) and plots them as three overlaid radar traces so a reviewer
    can see at a glance how this patient's values compare across categories."""
    categories = [f.replace("mean ", "").title() for f in feature_names[:10]]
    z = scaler.transform(np.array(inputs).reshape(1, -1))[0]

    def squash(v):  # squash z-scores into a readable 0-1 band for the radar
        return list(np.clip((np.array(v) + 3) / 6, 0, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=squash(z[0:10]), theta=categories, fill="toself", name="Mean"))
    fig.add_trace(go.Scatterpolar(r=squash(z[10:20]), theta=categories, fill="toself", name="Standard Error"))
    fig.add_trace(go.Scatterpolar(r=squash(z[20:30]), theta=categories, fill="toself", name="Worst"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        showlegend=True,
        height=430,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig

# ── Sidebar: patient input ───────────────────────────────────────────────
st.sidebar.header("Patient Data Input")
st.sidebar.caption("Enter clinical measurements (demo dataset: Breast Cancer Wisconsin features)")

if "review_log" not in st.session_state:
    st.session_state.review_log = []

inputs = []
with st.sidebar.form("patient_form"):
    for name in feature_names:
        val = st.number_input(name, value=0.0, format="%.4f", key=name)
        inputs.append(val)
    submitted = st.form_submit_button("Run Diagnosis")

# ── Main panel ────────────────────────────────────────────────────────────
st.title("🩺 Intelligent Medical Diagnosis Framework")
st.caption("MLP + Backpropagation classifier with Human-in-the-Loop validation")

col_chart, col1, col2 = st.columns([1.3, 1.4, 1])

if submitted:
    X = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob_benign = float(model.predict(X_scaled, verbose=0)[0][0])
    prob_malignant = 1 - prob_benign
    predicted_class = "Benign" if prob_benign >= 0.5 else "Malignant"
    confidence = max(prob_benign, prob_malignant)

    with col_chart:
        st.subheader("Patient Measurement Profile")
        st.plotly_chart(get_radar_chart(inputs, feature_names, scaler), use_container_width=True)

    with col1:
        st.subheader("Prediction Result")
        st.metric("Predicted Class", predicted_class)
        st.progress(confidence)
        st.write(f"Confidence: **{confidence*100:.1f}%**")
        st.write(f"P(benign) = {prob_benign:.3f} | P(malignant) = {prob_malignant:.3f}")

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                "⚠️ LOW CONFIDENCE — this case is flagged for manual clinical review "
                "rather than being auto-finalized."
            )
            if st.button("Send to Clinician for Review"):
                st.session_state.review_log.append({
                    "prediction": predicted_class,
                    "confidence": round(confidence, 3),
                    "status": "Pending Review",
                })
                st.success("Case sent to the review queue below.")
        else:
            st.success("✅ High-confidence prediction — no manual review required.")

    with col2:
        st.subheader("Decision Rule")
        st.write(f"Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
        st.write(
            "Cases below the threshold are never auto-finalized — this is the "
            "safety gate your project proposes in place of a fully autonomous "
            "black-box classifier."
        )

st.divider()
st.subheader("🧑‍⚕️ Human Review Queue")
if st.session_state.review_log:
    st.table(st.session_state.review_log)
else:
    st.caption("No cases currently pending review.")

st.divider()
with st.expander("About this system"):
    st.write(
        "This dashboard implements the framework proposed in your literature "
        "survey: a lightweight MLP classifier trained with backpropagation, "
        "paired with an interactive validation layer so low-confidence "
        "predictions are routed to a human clinician instead of being acted "
        "on automatically. This is a decision-support prototype for a course "
        "project, not a certified diagnostic device."
    )
