# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")

st.title("🎓 Student Pass / Fail Predictor")

# ==============================
# 2. LOAD & PREPARE DATA
# ==============================
df = pd.read_csv("pass_fail_dataset.csv")

df.fillna(df.mean(numeric_only=True), inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop('pass', axis=1)
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 3. TRAIN OR LOAD MODEL
# ==============================
model_file = "student_model.pkl"

if not os.path.exists(model_file):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
else:
    model = joblib.load(model_file)

# ==============================
# 4. USER INPUT
# ==============================
st.subheader("Enter Student Details")

study_hours = st.number_input("Study Hours", 0.0, 24.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0)
previous_grade = st.number_input("Previous Grade", 0.0, 100.0)

# ==============================
# 5. PREDICTION
# ==============================
if st.button("Predict Result"):

    try:
        features = np.array([[study_hours, attendance, previous_grade]])
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("✅ Student will PASS")
        else:
            st.error("❌ Student may FAIL")

    except Exception as e:
        st.error(f"Error: {e}")
