"""
Streamlit Human-in-the-Loop Multi-Disease Diagnosis Dashboard
Loads a trained MLP model (selected via a disease dropdown) and lets a user
enter patient values to get a prediction. Low-confidence predictions are
automatically flagged for manual clinical review.

RUN LOCALLY:
    pip install -r requirements.txt
    streamlit run app.py

DEPLOY (free): push this folder to a public GitHub repo, then go to
share.streamlit.io -> New app -> pick the repo -> set main file to app.py.

Required model files (per disease), all in the same folder as this script:
  Breast Cancer: mlp_diagnosis_model.keras, scaler.pkl, feature_names.pkl
  Diabetes:      diabetes_mlp_model.keras, diabetes_scaler.pkl, diabetes_feature_names.pkl
"""

import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import io
import re
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes

st.set_page_config(page_title="Intelligent Medical Diagnosis Dashboard", layout="wide")

CONFIDENCE_THRESHOLD = 0.75  # below this, route to human review

# ── Disease configuration ─────────────────────────────────────────────────
# Each entry defines everything the app needs to run that disease's model:
# which files to load, what the two output classes are called, which fields
# are legitimately allowed to be zero, and how to group the radar chart.
DISEASE_CONFIGS = {
    "Breast Cancer": {
        "model_file": "mlp_diagnosis_model.keras",
        "scaler_file": "scaler.pkl",
        "features_file": "feature_names.pkl",
        "class_1_label": "Benign",      # model output ~1 means this
        "class_0_label": "Malignant",   # model output ~0 means this
        "zero_valid_fields": set(),      # none of the 30 imaging features can legitimately be 0
        "radar_groups": [("Mean", 0, 10), ("Standard Error", 10, 20), ("Worst", 20, 30)],
        "input_caption": "Enter clinical measurements (demo dataset: Breast Cancer Wisconsin features)",
        "upload_caption": (
            "Only works if the report already lists these exact 30 measurements "
            "(e.g. printed from imaging software). A regular patient chart won't "
            "contain them, and nothing will auto-fill."
        ),
        "about": (
            "A lightweight MLP classifier trained with backpropagation on the "
            "Breast Cancer Wisconsin dataset, paired with an interactive "
            "validation layer so low-confidence predictions are routed to a "
            "human clinician instead of being acted on automatically."
        ),
    },
    "Diabetes": {
        "model_file": "diabetes_mlp_model.keras",
        "scaler_file": "diabetes_scaler.pkl",
        "features_file": "diabetes_feature_names.pkl",
        "class_1_label": "Diabetic",
        "class_0_label": "Non-Diabetic",
        "zero_valid_fields": {"Pregnancies"},  # 0 pregnancies is a real, valid value
        "radar_groups": None,  # only 8 features, shown as a single trace
        "input_caption": "Enter clinical measurements (Pima Indians Diabetes dataset features)",
        "upload_caption": (
            "Only works if the report already lists these exact 8 values "
            "(Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, "
            "BMI, Diabetes Pedigree Function, Age). A general chart won't "
            "contain them all, and nothing will auto-fill."
        ),
        "about": (
            "A lightweight MLP classifier trained with backpropagation on the "
            "Pima Indians Diabetes dataset, matching the patent draft's own "
            "Working Example 1. This dataset is noisier than the Breast Cancer "
            "one, so expect meaningfully lower accuracy — that reflects real "
            "clinical data quality, not a bug."
        ),
    },
}

# ── Sidebar: disease selector (drives everything else on the page) ───────
st.sidebar.header("Diagnosis Type")
disease = st.sidebar.selectbox("Select disease model", list(DISEASE_CONFIGS.keys()))
cfg = DISEASE_CONFIGS[disease]

# ── Load model + preprocessing artifacts for the selected disease ────────
@st.cache_resource
def load_artifacts(model_file, scaler_file, features_file):
    model = tf.keras.models.load_model(model_file)
    scaler = joblib.load(scaler_file)
    feature_names = joblib.load(features_file)
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts(cfg["model_file"], cfg["scaler_file"], cfg["features_file"])
except Exception:
    st.error(
        f"Model files for {disease} not found. Train the model first, then copy "
        f"{cfg['model_file']}, {cfg['scaler_file']}, and {cfg['features_file']} "
        f"into this folder."
    )
    st.stop()

# ── Report auto-fill: extract text from an uploaded PDF/image, then match ──
# ── it against the disease's known feature names to pre-fill the sidebar ──
def extract_text_from_upload(uploaded_file):
    """Returns raw text from a PDF (native text, falling back to OCR for
    scanned/image-based PDFs) or an image file."""
    file_bytes = uploaded_file.getvalue()
    text = ""

    if uploaded_file.type == "application/pdf":
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception:
            pass
        if len(text.strip()) < 20:
            try:
                images = convert_from_bytes(file_bytes)
                for img in images:
                    text += pytesseract.image_to_string(img) + "\n"
            except Exception:
                pass
    else:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(img)
        except Exception:
            pass

    return text


def parse_feature_values(text, feature_names):
    """Looks for each known feature name in the extracted text and captures
    the nearest following number as its value."""
    results = {}
    normalized = text.lower()
    for name in feature_names:
        pattern = re.escape(name.lower()).replace(r"\ ", r"\s+")
        pattern += r"[^0-9\-]{0,20}(-?\d+\.?\d*)"
        match = re.search(pattern, normalized)
        if match:
            try:
                results[name] = float(match.group(1))
            except ValueError:
                pass
    return results


# ── Radar chart: visualize the patient's profile ─────────────────────────
def get_radar_chart(inputs, feature_names, scaler, groups):
    """If `groups` is given (list of (label, start, end)), plots one trace
    per group (e.g. Mean / Standard Error / Worst) using the first group's
    slice for category labels. Otherwise plots every feature as one trace."""
    z = scaler.transform(np.array(inputs).reshape(1, -1))[0]

    def squash(v):
        return list(np.clip((np.array(v) + 3) / 6, 0, 1))

    fig = go.Figure()
    if groups:
        categories = [feature_names[groups[0][1]:groups[0][2]][i].replace("mean ", "").title()
                      for i in range(groups[0][2] - groups[0][1])]
        for label, start, end in groups:
            fig.add_trace(go.Scatterpolar(r=squash(z[start:end]), theta=categories, fill="toself", name=label))
    else:
        categories = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", f) for f in feature_names]
        fig.add_trace(go.Scatterpolar(r=squash(z), theta=categories, fill="toself", name="Patient Profile"))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        showlegend=True,
        height=430,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig

# ── Sidebar: patient input ───────────────────────────────────────────────
st.sidebar.header("Patient Data Input")
st.sidebar.caption(cfg["input_caption"])

if "review_log" not in st.session_state:
    st.session_state.review_log = []

st.sidebar.subheader("📄 Auto-fill from Report")
uploaded_report = st.sidebar.file_uploader(
    "Upload a lab report (PDF or photo)", type=["pdf", "png", "jpg", "jpeg"], key=f"uploader_{disease}"
)
st.sidebar.caption(cfg["upload_caption"])

if uploaded_report is not None:
    with st.spinner("Reading report..."):
        report_text = extract_text_from_upload(uploaded_report)
        extracted = parse_feature_values(report_text, feature_names)

    if extracted:
        for name, val in extracted.items():
            st.session_state[f"{disease}_{name}"] = val
        missing = [f for f in feature_names if f not in extracted]
        st.sidebar.success(f"Auto-filled {len(extracted)} of {len(feature_names)} fields — review below before running.")
        if missing:
            st.sidebar.warning(f"Not found, please fill manually: {', '.join(missing)}")
    else:
        st.sidebar.error(
            "Couldn't match any known fields in this document. Either the "
            "format doesn't align with the expected measurements, or the "
            "scan quality is too low for OCR — enter values manually below."
        )

inputs = []
with st.sidebar.form(f"patient_form_{disease}"):
    for name in feature_names:
        val = st.number_input(name, value=0.0, format="%.4f", key=f"{disease}_{name}")
        inputs.append(val)
    submitted = st.form_submit_button("Run Diagnosis")

# ── Main panel ────────────────────────────────────────────────────────────
st.title("🩺 Intelligent Medical Diagnosis Framework")
st.caption(f"MLP + Backpropagation classifier with Human-in-the-Loop validation — {disease} model")

col_chart, col1, col2 = st.columns([1.3, 1.4, 1])

if submitted:
    zero_fields = [name for name, val in zip(feature_names, inputs)
                   if val == 0.0 and name not in cfg["zero_valid_fields"]]

    if zero_fields:
        st.error(
            f"⚠️ Cannot run diagnosis — {len(zero_fields)} field(s) are still "
            f"at 0.0000, which usually means they were never filled in (most "
            f"clinical measurements are never exactly zero). Running with "
            f"missing data produces an unreliable, falsely confident result.\n\n"
            f"**Missing:** {', '.join(zero_fields)}\n\n"
            f"Fill these in manually, or upload a report that lists them, then "
            f"click Run Diagnosis again."
        )
        st.stop()

    X = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob_class1 = float(model.predict(X_scaled, verbose=0)[0][0])
    prob_class0 = 1 - prob_class1
    predicted_class = cfg["class_1_label"] if prob_class1 >= 0.5 else cfg["class_0_label"]
    confidence = max(prob_class1, prob_class0)

    with col_chart:
        st.subheader("Patient Measurement Profile")
        st.plotly_chart(get_radar_chart(inputs, feature_names, scaler, cfg["radar_groups"]), use_container_width=True)

    with col1:
        st.subheader("Prediction Result")
        st.metric("Predicted Class", predicted_class)
        st.progress(confidence)
        st.write(f"Confidence: **{confidence*100:.1f}%**")
        st.write(f"P({cfg['class_1_label']}) = {prob_class1:.3f} | P({cfg['class_0_label']}) = {prob_class0:.3f}")

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                "⚠️ LOW CONFIDENCE — this case is flagged for manual clinical review "
                "rather than being auto-finalized."
            )
            if st.button("Send to Clinician for Review"):
                st.session_state.review_log.append({
                    "disease": disease,
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
            "safety gate this project proposes in place of a fully autonomous "
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
    st.write(cfg["about"])
    st.write(
        "This is a decision-support prototype for a course project, not a "
        "certified diagnostic device."
    )
