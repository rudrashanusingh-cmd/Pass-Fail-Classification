import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODELS + COLUMNS
# ==============================
rf_model = joblib.load("rf_model.pkl")
dt_model = joblib.load("dt_model.pkl")
lr_model = joblib.load("lr_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Pass/Fail Predictor", layout="centered")

st.title("🎓 Student Pass/Fail Predictor")

# Model selection
model_choice = st.selectbox(
    "Select Model",
    ["Random Forest", "Decision Tree", "Logistic Regression"]
)

study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, step=1.0)

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):
    try:
        input_data = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_score": previous_score
        }])

        # Align columns
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Select model
        if model_choice == "Decision Tree":
            model = dt_model
        elif model_choice == "Logistic Regression":
            model = lr_model
        else:
            model = rf_model

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success(f"✅ PASS (Model: {model_choice})")
        else:
            st.error(f"❌ FAIL (Model: {model_choice})")

    except Exception as e:
        st.error(f"Error: {e}")
