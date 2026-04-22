# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ==============================
# 2. LOAD & PREPARE DATA
# ==============================
df = pd.read_csv("pass_fail_dataset_extended.csv")

df.fillna(df.mean(numeric_only=True), inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('pass', axis=1)
y = df['pass']

# 🔥 Save column order (VERY IMPORTANT)
model_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 3. TRAIN MODELS
# ==============================
rf_file = "rf_model.pkl"
dt_file = "dt_model.pkl"
lr_file = "lr_model.pkl"

# Random Forest
if not os.path.exists(rf_file):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, rf_file)
else:
    rf_model = joblib.load(rf_file)

# Decision Tree
if not os.path.exists(dt_file):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, dt_file)
else:
    dt_model = joblib.load(dt_file)

# Logistic Regression
if not os.path.exists(lr_file):
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, lr_file)
else:
    lr_model = joblib.load(lr_file)

# ==============================
# 4. CREATE FLASK APP
# ==============================
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        input_data = pd.DataFrame([{
            "study_hours": data["study_hours"],
            "attendance": data["attendance"],
            "previous_score": data["previous_score"]
        }])

        # 🔥 Align columns with training data
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        model_type = data.get("model", "rf")

        if model_type == "dt":
            prediction = dt_model.predict(input_data)
        elif model_type == "lr":
            prediction = lr_model.predict(input_data)
        else:
            prediction = rf_model.predict(input_data)

        return jsonify({
            "model_used": model_type,
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# 5. RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
