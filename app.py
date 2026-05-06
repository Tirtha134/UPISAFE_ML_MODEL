import pickle
import nltk
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
import requests

# =========================
# INIT
# =========================
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# GOOGLE DRIVE DIRECT LINK
# =========================
# Original:
# https://drive.google.com/file/d/1Bcgtz2R5orqaILgTmj2Ap4W0oXm_DZwP/view

MODEL_URL = "https://drive.google.com/uc?export=download&id=1Bcgtz2R5orqaILgTmj2Ap4W0oXm_DZwP"

MODEL_PATH = os.path.join(BASE_DIR, "fraud_detector.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "transaction_label.pkl")

# =========================
# DOWNLOAD MODEL IF NOT EXISTS
# =========================
def download_file(url, path):
    try:
        print(f"⬇️ Downloading model from Drive...")
        r = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024):
                if chunk:
                    f.write(chunk)
        print("✅ Download complete")
    except Exception as e:
        print("❌ Download failed:", e)

# Download only if not present
if not os.path.exists(MODEL_PATH):
    download_file(MODEL_URL, MODEL_PATH)

# =========================
# LOAD MODEL
# =========================
try:
    model_rfc = pickle.load(open(MODEL_PATH, "rb"))
    tr_label = pickle.load(open(LABEL_PATH, "rb"))
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Load Error:", e)
    model_rfc = None
    tr_label = None

# =========================
# HOME ROUTE
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Flask Running ✅",
        "model_loaded": model_rfc is not None
    })

# =========================
# TEXT CLEANING
# =========================
def transform_text(text):
    if not text:
        return ""
    text = text.lower()
    words = nltk.word_tokenize(text)
    return "".join([w for w in words if w.isalnum()])

# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model_rfc is None or tr_label is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(force=True)
        print("\n📥 INPUT:", data)

        # =========================
        # INPUT VALUES
        # =========================
        upi_id = data.get("upi_id", "")
        amount = float(data.get("amount", 0))
        date = data.get("date", "0/0/0")
        time = data.get("time", "00:00")
        type_input = data.get("type", "PAYMENT").upper()

        if type_input not in ["PAYMENT", "RECEIVE"]:
            type_input = "PAYMENT"

        # =========================
        # FEATURE ENGINEERING
        # =========================
        upi_mod = transform_text(upi_id)

        try:
            type_mod = tr_label.transform([type_input])[0]
        except:
            type_mod = 0

        try:
            date_mod = int(date.replace("/", ""))
        except:
            date_mod = 0

        try:
            time_mod = int(time.replace(":", ""))
        except:
            time_mod = 0

        df = pd.DataFrame([{
            "num_char_modified": len(upi_mod),
            "new_r_id": upi_mod,
            "Amount": amount,
            "Date_mod": date_mod,
            "Time_mod": time_mod,
            "Type_mod": type_mod
        }])

        print("📊 DF:\n", df)

        # =========================
        # PROBABILITY
        # =========================
        if hasattr(model_rfc, "predict_proba"):
            prob = float(model_rfc.predict_proba(df)[0][1])
        else:
            prob = 0.5

        print("🔍 PROBABILITY:", prob)

        risk_score = round(prob * 100, 2)

        # =========================
        # FINAL DECISION
        # =========================
        pred = 1 if risk_score >= 50 else 0

        if risk_score >= 70:
            risk_level = "HIGH 🚨"
        elif risk_score >= 50:
            risk_level = "MEDIUM ⚠️"
        else:
            risk_level = "LOW ✅"

        print("🤖 FINAL:", pred, risk_score)

        return jsonify({
            "Fraud_Result": bool(pred == 1),
            "Risk_Score": risk_score,
            "Risk_Level": risk_level,
            "Transaction_Type": type_input
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)