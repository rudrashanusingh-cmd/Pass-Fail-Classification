# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# ==============================
# 2. LOAD / TRAIN MODEL
# ==============================
model_file = "student_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        # Train model if not exists
        df = pd.read_csv("student_data.csv")

        df.fillna(df.mean(numeric_only=True), inplace=True)
        df = pd.get_dummies(df, drop_first=True)

        X = df.drop('pass', axis=1)
        y = df['pass']

        model = RandomForestClassifier()
        model.fit(X, y)

        joblib.dump(model, model_file)
        return model

model = load_model()

# ==============================
# 3. STREAMLIT UI
# ==============================
st.title("🎓 Student Pass Prediction App")

st.write("Enter student details:")

study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
previous_grade = st.number_input("Previous Grade", min_value=0.0, max_value=100.0, step=1.0)

# ==============================
# 4. PREDICTION
# ==============================
if st.button("Predict"):
    try:
        features = np.array([[study_hours, attendance, previous_grade]])
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("✅ Student will PASS")
        else:
            st.error("❌ Student will FAIL")

    except Exception as e:
        st.error(f"Error: {e}")
