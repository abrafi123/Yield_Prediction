from flask import Flask, render_template, request
import pandas as pd
import pickle
from pathlib import Path

# Flask app তৈরি
app = Flask(__name__)

# === মডেল লোড করা ===
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "dtr.pkl", "rb") as f:
    dtr = pickle.load(f)

with open(BASE_DIR / "preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# === ইনপুট কলাম ===
INPUT_COLS = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "Area",
    "Item"
]

@app.route("/")
def index():
    # শুধু ফর্ম দেখাবে, কোনো প্রেডিকশন নয়
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ফর্ম ডেটা নেওয়া
        row = {
            "Year": int(request.form["Year"]),
            "average_rain_fall_mm_per_year": float(request.form["average_rain_fall_mm_per_year"]),
            "pesticides_tonnes": float(request.form["pesticides_tonnes"]),
            "avg_temp": float(request.form["avg_temp"]),
            "Area": request.form["Area"].strip(),
            "Item": request.form["Item"].strip(),
        }

        # ডেটাফ্রেম বানানো
        df = pd.DataFrame([row], columns=INPUT_COLS)

        # transform + predict
        Xt = preprocessor.transform(df)
        yhat = float(dtr.predict(Xt)[0])

        return render_template("index.html", pred=round(yhat, 2))
    except Exception as e:
        return render_template("index.html", pred=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)