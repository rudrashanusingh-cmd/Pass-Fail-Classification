# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ==============================
# 2. LOAD & PREPARE DATA
# ==============================
df = pd.read_csv("student_data.csv")

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop('pass', axis=1)
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 3. TRAIN MODEL (only once)
# ==============================
model_file = "student_model.pkl"

if not os.path.exists(model_file):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    print("Model trained and saved!")
else:
    model = joblib.load(model_file)
    print("Model loaded!")

# ==============================
# 4. CREATE FLASK APP
# ==============================
app = Flask(__name__)

@app.route("/")
def home():
    return "Student Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Input example: [study_hours, attendance, previous_grade]
        features = np.array(data['features']).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# 5. RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)