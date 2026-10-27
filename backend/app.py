"""Flask backend: dataset se model train karta hai aur /api/predict deta hai."""
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

filename = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

possible_paths = [
    ROOT / "Datasets" / filename,
    ROOT / "backend" / "Datasets" / filename,
    Path(__file__).resolve().parent / "Datasets" / filename,
]

DATA = next((p for p in possible_paths if p.is_file()), possible_paths[0])

print("Project root:", ROOT)
print("Dataset path:", DATA)
print("Dataset exists:", DATA.is_file())
app = Flask(__name__, static_folder=None)
CORS(app)

NUM = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CAT = ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity",
       "TechSupport", "PaperlessBilling"]

print("ROOT:", ROOT)
print("DATA:", DATA)
print("DATA EXISTS:", DATA.exists())
print("DATASETS EXISTS:", DATA.parent.exists())
def train():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    y = (df["Churn"] == "Yes").astype(int)
    X = df[NUM + CAT]

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    model.fit(X, y)
    return model


MODEL = train()


@app.route("/")
def home():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.get_json(force=True)
    tenure = float(d["tenure"])
    monthly = float(d["MonthlyCharges"])
    row = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
        "SeniorCitizen": int(d.get("SeniorCitizen", 0)),
        "Contract": d["Contract"],
        "InternetService": d["InternetService"],
        "PaymentMethod": d["PaymentMethod"],
        "OnlineSecurity": d["OnlineSecurity"],
        "TechSupport": d["TechSupport"],
        "PaperlessBilling": d["PaperlessBilling"],
    }])
    p = float(MODEL.predict_proba(row)[0, 1])
    level = "High" if p >= 0.6 else "Medium" if p >= 0.35 else "Low"
    return jsonify({"probability": round(p, 3), "risk": level})


@app.route("/<path:name>")
def assets(name):
    return send_from_directory(FRONTEND, name)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
