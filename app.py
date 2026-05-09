import os
import pickle
import gdown
import nltk
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# INIT
# =========================
nltk.download("punkt", quiet=True)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# FILE PATHS
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "fraud_detector.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "transaction_label.pkl")

# =========================
# GOOGLE DRIVE FILE IDS
# =========================
MODEL_FILE_ID = "1Bcgtz2R5orqaILgTmj2Ap4W0oXm_DZwP"
LABEL_FILE_ID = "1TO9egtEk7ETmQqfiZ-ArJ0rgV1Ufzq1e"

# =========================
# DOWNLOAD FILES
# =========================
def download_file(file_id, output_path, file_name):

    if os.path.exists(output_path):
        print(f"✅ {file_name} already exists")
        return True

    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        print(f"⬇️ Downloading {file_name}...")
        gdown.download(url=url, output=output_path, quiet=False)
        print(f"✅ {file_name} downloaded successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to download {file_name}: {e}")
        return False


# =========================
# DOWNLOAD MODELS
# =========================
download_file(MODEL_FILE_ID, MODEL_PATH, "fraud_detector.pkl")
download_file(LABEL_FILE_ID, LABEL_PATH, "transaction_label.pkl")


# =========================
# LOAD MODEL
# =========================
try:
    with open(MODEL_PATH, "rb") as f:
        model_rfc = pickle.load(f)

    with open(LABEL_PATH, "rb") as f:
        tr_label = pickle.load(f)

    print("✅ Models Loaded Successfully")

    if hasattr(model_rfc, "feature_names_in_"):
        print("📋 Expected features:", list(model_rfc.feature_names_in_))
    else:
        print("⚠️  n_features_in_:", getattr(model_rfc, "n_features_in_", "unknown"))

except Exception as e:
    print("❌ Model Load Error:", e)
    model_rfc = None
    tr_label  = None


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
# HEALTH ROUTE
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_rfc is not None,
        "expected_features": (
            list(model_rfc.feature_names_in_)
            if model_rfc is not None and hasattr(model_rfc, "feature_names_in_")
            else "unknown"
        )
    })


# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    try:
        words = nltk.word_tokenize(text)
        return "".join([w for w in words if w.isalnum()])
    except Exception:
        return "".join([c for c in text if c.isalnum()])


# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        if model_rfc is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        print("\n📥 INPUT:", data)

        # =========================
        # PARSE INPUTS
        # =========================
        upi_id     = str(data.get("upi_id", ""))
        amount     = float(data.get("amount", 0))
        date       = str(data.get("date", "2026-01-01"))
        time_val   = str(data.get("time", "00:00"))
        type_input = str(data.get("type", "PAYMENT")).upper().strip()

        if type_input not in ["PAYMENT", "RECEIVE"]:
            type_input = "PAYMENT"

        # =========================
        # FEATURE ENGINEERING
        # =========================

        # Clean UPI ID string — kept as STRING because the model pipeline
        # has a text transformer that expects 'new_r_id' as a string column.
        # Expected features confirmed: ['num_char_modified', 'new_r_id',
        #                               'Amount', 'Date_mod', 'Time_mod', 'Type_mod']
        new_r_id          = clean_text(upi_id)
        num_char_modified = len(new_r_id)

        # Type encoding: PAYMENT=0, RECEIVE=1
        Type_mod = 0 if type_input == "PAYMENT" else 1

        # Date: "2026-05-12" or "12/05/2026" → integer 20260512
        try:
            Date_mod = int(date.replace("-", "").replace("/", ""))
        except Exception:
            Date_mod = 0

        # Time: "14:35" → integer 1435
        try:
            Time_mod = int(time_val.replace(":", ""))
        except Exception:
            Time_mod = 0

        # =========================
        # BUILD DATAFRAME
        # Column order matches model's feature_names_in_ exactly:
        # ['num_char_modified', 'new_r_id', 'Amount', 'Date_mod', 'Time_mod', 'Type_mod']
        # =========================
        df = pd.DataFrame([{
            "num_char_modified": num_char_modified,
            "new_r_id":          new_r_id,
            "Amount":            amount,
            "Date_mod":          Date_mod,
            "Time_mod":          Time_mod,
            "Type_mod":          Type_mod,
        }])

        print("\n📊 FEATURES:")
        print(df)

        # =========================
        # PREDICT
        # =========================
        if hasattr(model_rfc, "predict_proba"):
            proba      = model_rfc.predict_proba(df)
            fraud_prob = float(proba[0][1]) if proba.shape[1] >= 2 else float(proba[0][0])
        else:
            fraud_prob = float(model_rfc.predict(df)[0])

        # =========================
        # RISK SCORE & LEVEL
        # =========================
        risk_score   = round(fraud_prob * 100, 2)
        fraud_result = risk_score >= 50

        if risk_score >= 70:
            risk_level = "HIGH 🚨"
        elif risk_score >= 50:
            risk_level = "MEDIUM ⚠️"
        else:
            risk_level = "LOW ✅"

        print(f"\n🤖 {'FRAUD' if fraud_result else 'SAFE'} | Score: {risk_score} | {risk_level}")

        return jsonify({
            "Fraud_Result":     fraud_result,
            "Risk_Score":       risk_score,
            "Risk_Level":       risk_level,
            "Transaction_Type": type_input
        })

    except Exception as e:
        import traceback
        print("❌ PREDICT ERROR:\n", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    port  = int(os.getenv("FLASK_PORT", 8000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)