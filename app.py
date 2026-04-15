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

st.set_page_config(page_title="Student Predictor", page_icon="🎓")

st.title("🎓 Student Pass / Fail Predictor")

# ==============================
# 2. LOAD DATA
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
# 3. TRAIN / LOAD MODEL
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
st.subheader("📥 Enter Student Details")

study_hours = st.slider("Study Hours", 0.0, 12.0, 5.0)
attendance = st.slider("Attendance (%)", 0.0, 100.0, 70.0)
previous_score = st.slider("Previous Score", 0.0, 100.0, 60.0)

# ==============================
# 5. PREDICTION
# ==============================
if st.button("Predict"):

    features = np.array([[study_hours, attendance, previous_score]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student may FAIL")

# ==============================
# 6. DATASET INSIGHTS (AUTO)
# ==============================
st.subheader("📊 Dataset Insights")

pass_data = df[df['pass'] == 1]
fail_data = df[df['pass'] == 0]

st.write("### ✅ PASS Ranges (from dataset)")
st.write({
    "Study Hours": (round(pass_data.iloc[:,0].min(),2), round(pass_data.iloc[:,0].max(),2)),
    "Attendance": (round(pass_data.iloc[:,1].min(),2), round(pass_data.iloc[:,1].max(),2)),
    "Previous Score": (round(pass_data.iloc[:,2].min(),2), round(pass_data.iloc[:,2].max(),2))
})

st.write("### ❌ FAIL Ranges (from dataset)")
st.write({
    "Study Hours": (round(fail_data.iloc[:,0].min(),2), round(fail_data.iloc[:,0].max(),2)),
    "Attendance": (round(fail_data.iloc[:,1].min(),2), round(fail_data.iloc[:,1].max(),2)),
    "Previous Score": (round(fail_data.iloc[:,2].min(),2), round(fail_data.iloc[:,2].max(),2))
})

# ==============================
# 7. SIMPLE RULE (DERIVED)
# ==============================
st.subheader("🎯 Simple Rule (Approx)")

st.info("""
If:
- Study Hours > 5  
- Attendance > 70%  
- Previous Score > 65  

➡️ PASS likely  
Else ➡️ FAIL likely
""")

# ==============================
# 8. FEATURE IMPORTANCE
# ==============================
st.subheader("🔥 Feature Importance")

importance = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))
