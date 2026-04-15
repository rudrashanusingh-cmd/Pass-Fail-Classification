# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import streamlit as st
import numpy as np
import joblib

# ==============================
# 2. LOAD MODELS
# ==============================
rf_model = joblib.load("rf_model.pkl")
dt_model = joblib.load("dt_model.pkl")
lr_model = joblib.load("lr_model.pkl")

# ==============================
# 3. STREAMLIT UI
# ==============================
st.set_page_config(page_title="Student Pass/Fail Predictor", layout="centered")

st.title("🎓 Student Pass/Fail Prediction")
st.write("Enter student details to predict result")

# Example inputs (change according to your dataset columns)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Model selection
model_choice = st.selectbox(
    "Choose Model",
    ["Random Forest", "Decision Tree", "Logistic Regression"]
)

# ==============================
# 4. PREDICTION
# ==============================
if st.button("Predict"):
    features = np.array([[feature1, feature2, feature3]])

    if model_choice == "Decision Tree":
        prediction = dt_model.predict(features)
        model_used = "Decision Tree"
    elif model_choice == "Logistic Regression":
        prediction = lr_model.predict(features)
        model_used = "Logistic Regression"
    else:
        prediction = rf_model.predict(features)
        model_used = "Random Forest"

    result = "Pass ✅" if prediction[0] == 1 else "Fail ❌"

    st.subheader(f"Result: {result}")
    st.write(f"Model Used: {model_used}")

# ==============================
# 5. FOOTER
# ==============================
st.markdown("---")
st.write("Made with Streamlit")
