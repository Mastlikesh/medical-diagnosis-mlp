# Intelligent Medical Diagnosis Framework — Build & Deploy Guide

This turns your PBL slides + literature survey into a working, demoable project.
Total hands-on time: roughly 2–4 hours if you follow the steps in order.

## What you're actually building
1. An **MLP neural network trained with backpropagation** (`train_model.py`) that
   classifies patient data — this satisfies your Objectives and Solutions slides.
2. A **Streamlit "Human-in-the-Loop" dashboard** (`app.py`) that flags low-confidence
   predictions for manual review — this directly answers the 3 gaps your own
   literature survey identifies (black-box isolation, no validation interface,
   no triage action). It also renders a radar chart of the patient's Mean /
   Standard Error / Worst measurements alongside the prediction, for a clearer
   visual demo.

## Platforms to use (and why)

| Step | Platform | Why |
|---|---|---|
| Train the model | **Google Colab** (colab.research.google.com) | Free, TensorFlow pre-installed, matches your Procurement Status slide exactly, no local setup |
| Store code | **GitHub** (github.com) | Free repo, required for Streamlit Cloud deployment, also doubles as your submission artifact |
| Host the dashboard | **Streamlit Community Cloud** (share.streamlit.io) | Free, deploys straight from GitHub, gives you a live public link to show in your review/demo |

## Step-by-step

### 1. Train the model (Google Colab) — ~30–45 min
1. Go to colab.research.google.com → New Notebook.
2. Copy the contents of `train_model.py` into cells (split at the `# ──` section
   markers if you want separate cells; it also just runs fine as one cell).
3. Runtime → Run all.
4. Read the printed accuracy/precision/recall/F1/ROC-AUC and look at the
   training curve plot — use these numbers directly in your Outcomes/Results slide.
5. In the Colab file browser (folder icon, left side), download:
   `mlp_diagnosis_model.keras`, `scaler.pkl`, `feature_names.pkl`.

### 2. Put the project on GitHub — ~15 min
1. Create a new **public** repo, e.g. `medical-mlp-diagnosis`.
2. Upload: `train_model.py`, `app.py`, `requirements.txt`, `README.md`, and the
   3 files you downloaded from Colab (model, scaler, feature names).
3. Commit.

### 3. Deploy the dashboard (Streamlit Community Cloud) — ~10 min
1. Go to share.streamlit.io → sign in with GitHub → "New app".
2. Pick your repo, branch `main`, main file path `app.py`.
3. Deploy. First build takes a few minutes (installing TensorFlow).
4. You'll get a public URL — this is what you demo in your PBL review.

### 4. (Optional) Test locally first
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Mapping this back to your slides
- **Objectives / Solutions slides** → satisfied by `train_model.py` (MLP + backprop,
  preprocessing, class-weighting for imbalance, dropout to reduce overfitting).
- **Hardware Development Status** → stays "no hardware required," this confirms it.
- **Project Timeline (Weeks 4–6)** → Week 4 = this training script, Week 5 =
  the run in Colab + tuning, Week 6 = the Streamlit dashboard + writeup.
- **Gap Identification (survey)** → satisfied by the confidence-threshold triage
  logic in `app.py` (low-confidence cases routed to a human review queue instead
  of an autonomous decision).

## Swapping the demo dataset for a "real" one later
`train_model.py` uses the Breast Cancer Wisconsin dataset built into scikit-learn
(the same dataset Wolberg & Mangasarian used, cited as [16] in your survey) so
you can get a working pipeline with zero download/auth hassle. Once this runs
end-to-end, you can swap in a UCI dataset that matches your other citations, e.g.:
- Pima Indians Diabetes dataset (matches Taha et al. [18])
- Cleveland Heart Disease dataset (matches Detrano et al. [19])

Both are on the UCI Machine Learning Repository and Kaggle. Only Section 2 of
`train_model.py` (data loading) needs to change — everything after it (model,
training loop, saving) works as-is as long as `y` is 0/1 encoded.

## Honest scope note for your review
This is a decision-support prototype for a course project, not a certified
medical device — say that explicitly in your presentation/demo. Reviewers will
respect that framing more than an overclaim.
