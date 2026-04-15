import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("student_model.pkl")

st.set_page_config(page_title="Student Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")

# ======================
# INPUTS (UPDATED)
# ======================
age = st.number_input("Age", 10, 100)
study_hours = st.number_input("Study Hours", 0.0, 24.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0)
previous_score = st.number_input("Previous Score", 0.0, 100.0)

# ======================
# PREDICT
# ======================
if st.button("Predict"):
    try:
        features = np.array([[age, study_hours, attendance, previous_score]])
        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("✅ Pass")
        else:
            st.error("❌ Fail")

    except Exception as e:
        st.error(f"Error: {e}")


