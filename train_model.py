"""
Intelligent Medical Diagnosis Framework
MLP Neural Network trained with Backpropagation (TensorFlow/Keras)

RUN THIS IN GOOGLE COLAB (matches your Procurement Status slide: Python + TensorFlow + Colab).
Just upload this .py as a notebook, or paste each section into its own cell.

Dataset: Breast Cancer Wisconsin (Diagnostic) dataset — the same dataset used in
Wolberg & Mangasarian [16] in your literature survey. It's built into scikit-learn,
so there's no download/auth step needed, and 569 patient records with 30 clinical
features is a realistic size for a first working model. Swap in a different UCI
dataset later (Section 7 below) once this pipeline works end-to-end.
"""

# ── 1. Install / Imports ────────────────────────────────────────────────
# In Colab: TensorFlow and scikit-learn are pre-installed. Just run this cell.
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import joblib

tf.random.set_seed(42)
np.random.seed(42)

# ── 2. Load Data ─────────────────────────────────────────────────────────
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = data.feature_names

print("Dataset shape:", X.shape)
print("Class distribution:", np.bincount(y))  # check imbalance

# ── 3. Preprocessing (cleaning, normalization, feature prep) ────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance (mirrors the SMOTE/class-weighting discussion, [24])
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# ── 4. Build the MLP (Input -> Hidden Layers -> Output) ─────────────────
def build_mlp(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),          # reduces overfitting, Srivastava et al. [25]
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),  # binary output
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Kingma & Ba [14]
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model

model = build_mlp(X_train_scaled.shape[1])
model.summary()

# ── 5. Train with Backpropagation ────────────────────────────────────────
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1,
)

# ── 6. Evaluate ───────────────────────────────────────────────────────────
y_probs = model.predict(X_test_scaled).flatten()
y_pred = (y_probs >= 0.5).astype(int)

print("\n=== Evaluation on Test Set ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_probs))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["malignant", "benign"]))

# Plot training curves — save + show
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="train_loss")
axes[0].plot(history.history["val_loss"], label="val_loss")
axes[0].set_title("Loss over epochs (Backpropagation convergence)")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="train_acc")
axes[1].plot(history.history["val_accuracy"], label="val_acc")
axes[1].set_title("Accuracy over epochs")
axes[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# ── 7. Save the model + scaler for the Streamlit dashboard ──────────────
model.save("mlp_diagnosis_model.keras")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(feature_names), "feature_names.pkl")
print("\nSaved: mlp_diagnosis_model.keras, scaler.pkl, feature_names.pkl")
print("Download these three files from the Colab file browser (left sidebar)")
print("and place them in the same folder as app.py before running the dashboard.")

# ── 8. (Optional) Swap in a different dataset later ─────────────────────
# To use a different UCI dataset (e.g. Pima Diabetes, Heart Disease/Cleveland),
# just replace Section 2 with a pandas.read_csv() of that dataset and make
# sure y is 0/1 encoded — everything else in this script stays the same.
