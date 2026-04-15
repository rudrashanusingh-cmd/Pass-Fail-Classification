import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# LOAD MODEL
# ======================
model = joblib.load("student_model.pkl")

st.set_page_config(page_title="Student Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")

# ======================
# INPUTS
# ======================
study_hours = st.number_input("Study Hours", 0.0, 24.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0)
previous_grade = st.number_input("Previous Grade", 0.0, 100.0)
age = st.number_input("Age", 0.0, 100.0)

# ======================
# PREDICT
# ======================
if st.button("Predict"):
    try:
        features = np.array([[study_hours, attendance, previous_grade]])
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("✅ Pass")
        else:
            st.error("❌ Fail")

    except Exception as e:
        st.error(f"Error: {e}")
