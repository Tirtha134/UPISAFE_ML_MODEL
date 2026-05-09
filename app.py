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
# NLTK
# =========================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# =========================
# INIT APP
# =========================
app = Flask(__name__)

# IMPORTANT FOR DEPLOYMENT
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True
)

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
# DOWNLOAD FUNCTION
# =========================
def download_file(file_id, output_path, file_name):

    if os.path.exists(output_path):
        print(f"✅ {file_name} already exists")
        return

    try:
        url = f"https://drive.google.com/uc?id={file_id}"

        print(f"⬇️ Downloading {file_name}...")

        gdown.download(
            url,
            output_path,
            quiet=False
        )

        print(f"✅ {file_name} downloaded")

    except Exception as e:
        print(f"❌ Download failed: {e}")


# =========================
# DOWNLOAD MODEL FILES
# =========================
download_file(MODEL_FILE_ID, MODEL_PATH, "fraud_detector.pkl")
download_file(LABEL_FILE_ID, LABEL_PATH, "transaction_label.pkl")

# =========================
# LOAD MODEL
# =========================
model_rfc = None
tr_label = None

try:

    with open(MODEL_PATH, "rb") as f:
        model_rfc = pickle.load(f)

    with open(LABEL_PATH, "rb") as f:
        tr_label = pickle.load(f)

    print("✅ MODEL LOADED")

    if hasattr(model_rfc, "feature_names_in_"):
        print("📋 FEATURES:")
        print(model_rfc.feature_names_in_)

except Exception as e:
    print("❌ MODEL LOAD ERROR")
    print(e)

# =========================
# HOME ROUTE
# =========================
@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "running",
        "model_loaded": model_rfc is not None
    })

# =========================
# HEALTH ROUTE
# =========================
@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": model_rfc is not None
    })

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):

    if not text:
        return ""

    text = str(text).lower().strip()

    try:
        words = nltk.word_tokenize(text)

        cleaned = " ".join([
            w for w in words if w.isalnum()
        ])

        return cleaned

    except Exception:

        cleaned = "".join([
            c for c in text if c.isalnum()
        ])

        return cleaned

# =========================
# SAFE FLOAT
# =========================
def safe_float(value, default=0):

    try:
        return float(value)
    except:
        return default

# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =========================
        # MODEL CHECK
        # =========================
        if model_rfc is None:
            return jsonify({
                "error": "Model not loaded"
            }), 500

        # =========================
        # GET JSON
        # =========================
        data = request.get_json()

        print("\n📥 RECEIVED:")
        print(data)

        if not data:
            return jsonify({
                "error": "No JSON received"
            }), 400

        # =========================
        # INPUTS
        # =========================
        upi_id = str(data.get("upi_id", "")).strip()

        amount = safe_float(
            data.get("amount", 0)
        )

        date = str(
            data.get("date", "")
        ).strip()

        time_val = str(
            data.get("time", "")
        ).strip()

        type_input = str(
            data.get("type", "PAYMENT")
        ).upper().strip()

        # =========================
        # VALIDATION
        # =========================
        if not upi_id:
            return jsonify({
                "error": "UPI ID required"
            }), 400

        if type_input not in ["PAYMENT", "RECEIVE"]:
            type_input = "PAYMENT"

        # =========================
        # FEATURE ENGINEERING
        # =========================

        # CLEAN TEXT
        new_r_id = clean_text(upi_id)

        # LENGTH
        num_char_modified = len(new_r_id)

        # AMOUNT
        Amount = amount

        # DATE FEATURE
        # Example: 2026-05-09 -> 9
        try:
            Date_mod = int(date.split("-")[2])
        except:
            Date_mod = 1

        # TIME FEATURE
        # Example: 14:35 -> 14
        try:
            Time_mod = int(time_val.split(":")[0])
        except:
            Time_mod = 0

        # TYPE
        Type_mod = 0 if type_input == "PAYMENT" else 1

        # =========================
        # DATAFRAME
        # =========================
        df = pd.DataFrame([{
            "num_char_modified": num_char_modified,
            "new_r_id": new_r_id,
            "Amount": Amount,
            "Date_mod": Date_mod,
            "Time_mod": Time_mod,
            "Type_mod": Type_mod
        }])

        print("\n📊 DATAFRAME")
        print(df)

        # =========================
        # PREDICTION
        # =========================
        prediction = model_rfc.predict(df)[0]

        if hasattr(model_rfc, "predict_proba"):

            probabilities = model_rfc.predict_proba(df)

            print("\n📈 PROBABILITIES")
            print(probabilities)

            fraud_prob = float(probabilities[0][1])

        else:
            fraud_prob = float(prediction)

        # =========================
        # RISK SCORE
        # =========================
        risk_score = round(fraud_prob * 100, 2)

        fraud_result = risk_score >= 50

        # =========================
        # RISK LEVEL
        # =========================
        if risk_score >= 70:
            risk_level = "HIGH 🚨"

        elif risk_score >= 50:
            risk_level = "MEDIUM ⚠️"

        else:
            risk_level = "LOW ✅"

        # =========================
        # FINAL RESPONSE
        # =========================
        response = {
            "Fraud_Result": fraud_result,
            "Risk_Score": risk_score,
            "Risk_Level": risk_level,
            "Transaction_Type": type_input
        }

        print("\n✅ RESPONSE")
        print(response)

        return jsonify(response)

    except Exception as e:

        import traceback

        print("\n❌ ERROR")
        print(traceback.format_exc())

        return jsonify({
            "error": str(e)
        }), 500

# =========================
# RUN
# =========================
if __name__ == "__main__":

    port = int(
        os.getenv("FLASK_PORT", 8000)
    )

    debug = os.getenv(
        "FLASK_DEBUG",
        "False"
    ).lower() == "true"

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug
    )
