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
# 2. PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Student Predictor",
    page_icon="🎓",
    layout="wide"
)

# ==============================
# 3. LOAD / TRAIN MODEL
# ==============================
model_file = "student_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
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
# 4. SIDEBAR
# ==============================
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Adjust input values")

study_hours = st.sidebar.slider("📘 Study Hours", 0.0, 24.0, 5.0)
attendance = st.sidebar.slider("📅 Attendance (%)", 0.0, 100.0, 75.0)
previous_grade = st.sidebar.slider("📊 Previous Grade", 0.0, 100.0, 60.0)

# ==============================
# 5. MAIN UI
# ==============================
st.title("🎓 Student Pass Prediction Dashboard")
st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Study Hours", study_hours)
col2.metric("Attendance", f"{attendance}%")
col3.metric("Previous Grade", previous_grade)

st.markdown("---")

# ==============================
# 6. PREDICTION BUTTON
# ==============================
if st.button("🚀 Predict Result"):
    try:
        features = np.array([[study_hours, attendance, previous_grade]])
        prediction = model.predict(features)

        st.markdown("## 🔍 Prediction Result")

        if prediction[0] == 1:
            st.success("✅ Student will PASS")
            st.balloons()
        else:
            st.error("❌ Student will FAIL")

        # Simple visualization
        st.markdown("### 📊 Input Summary")
        chart_data = pd.DataFrame({
            "Feature": ["Study Hours", "Attendance", "Previous Grade"],
            "Value": [study_hours, attendance, previous_grade]
        })
        st.bar_chart(chart_data.set_index("Feature"))

    except Exception as e:
        st.error(f"Error: {e}")

# ==============================
# 7. FOOTER
# ==============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
