# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

st.set_page_config(page_title="Student Predictor", page_icon="🎓")

st.title("🎓 Student Pass / Fail Predictor")

# ==============================
# 2. LOAD DATASET
# ==============================
df = pd.read_csv("pass_fail_dataset.csv")

df.fillna(df.mean(numeric_only=True), inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('pass', axis=1)
y = df['pass']

# ==============================
# 3. TRAIN / LOAD MODEL
# ==============================
model_file = "student_model.pkl"

if not os.path.exists(model_file):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_file)
else:
    model = joblib.load(model_file)

# ==============================
# 4. CALCULATE THRESHOLDS
# ==============================
pass_data = df[df['pass'] == 1]
fail_data = df[df['pass'] == 0]

cols = X.columns

# Dynamic thresholds for all features
thresholds = {}
for col in cols:
    thresholds[col] = (pass_data[col].min() + fail_data[col].max()) / 2

# ==============================
# 5. USER INPUT (NOW WITH AGE)
# ==============================
st.subheader("📥 Enter Details")

inputs = {}

for col in cols:
    if "age" in col.lower():
        inputs[col] = st.slider("Age", 10, 100, int(thresholds[col]))
    elif "study" in col.lower():
        inputs[col] = st.slider("Study Hours", 0.0, 12.0, float(thresholds[col]))
    elif "att" in col.lower():
        inputs[col] = st.slider("Attendance (%)", 0.0, 100.0, float(thresholds[col]))
    elif "score" in col.lower() or "grade" in col.lower():
        inputs[col] = st.slider("Previous Score", 0.0, 100.0, float(thresholds[col]))
    else:
        inputs[col] = st.number_input(col, value=float(thresholds[col]))

# ==============================
# 6. PREDICTION
# ==============================
if st.button("Predict"):

    feature_values = np.array([list(inputs.values())])

    # Model prediction
    model_pred = model.predict(feature_values)[0]

    # Rule-based prediction
    rule_pred = 1
    for col in cols:
        if inputs[col] < thresholds[col]:
            rule_pred = 0
            break

    st.subheader("📊 Result")

    if model_pred == 1:
        st.success("✅ Model: PASS")
    else:
        st.error("❌ Model: FAIL")

    if rule_pred == 1:
        st.info("🎯 Rule: PASS")
    # else:
    #     st.warning("⚠️ Rule: FAIL")

# ==============================
# 7. SHOW RULE
# ==============================
st.subheader("🎯 Dataset Rule")

for col in cols:
    st.write(f"{col} ≥ {round(thresholds[col],2)}")

# ==============================
# 8. FEATURE IMPORTANCE
# ==============================
st.subheader("🔥 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))
