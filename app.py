import os
import pickle
import gdown
import nltk
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

FLASK_PORT = int(os.environ.get("FLASK_PORT", 8000))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")

# -----------------------------
# Download NLTK Resources
# -----------------------------
nltk.download("punkt")
nltk.download("punkt_tab")

# -----------------------------
# Google Drive File IDs
# -----------------------------
MODEL_FILE_ID = "1Bcgtz2R5orqaILgTmj2Ap4W0oXm_DZwP"                               
LABEL_FILE_ID = "1TO9egtEk7ETmQqfiZ-ArJ0rgV1Ufzq1e"   

MODEL_PATH = "fraud_detector.pkl"
LABEL_PATH = "transaction_label.pkl"


# -----------------------------
# Download File If Not Exists
# -----------------------------
def download_if_not_exists(file_path, file_id):
    """
    Downloads a file from Google Drive only if it
    doesn't already exist locally.
    """
    if os.path.exists(file_path):
        print(f"Using existing file: {file_path}")
        return

    print(f"{file_path} not found.")
    print("Downloading from Google Drive...")

    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        gdown.download(url, file_path, quiet=False)

        if os.path.exists(file_path):
            print(f"Downloaded successfully: {file_path}")
        else:
            raise Exception("Download failed.")

    except Exception as e:
        raise Exception(f"Unable to download {file_path}: {e}")


# -----------------------------
# Download Model Files
# -----------------------------
download_if_not_exists(MODEL_PATH, MODEL_FILE_ID)
download_if_not_exists(LABEL_PATH, LABEL_FILE_ID)

# -----------------------------
# Load Pickle Files
# -----------------------------
with open(MODEL_PATH, "rb") as f:
    model_rfc = pickle.load(f)

with open(LABEL_PATH, "rb") as f:
    tr_label = pickle.load(f)

print("Model Loaded Successfully.")

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)


@app.route("/predict", methods=["GET"])
def home():
    return jsonify({
        "Request_Result": "Connected",
        "Next_Actions": "Get Data from Users"
    })


# -----------------------------
# Text Processing
# -----------------------------
def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    filtered = []

    for word in words:
        if word.isalnum():
            filtered.append(word)

    return "".join(filtered)


# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    upi_id = request.form.get("upi_id")
    date_input = request.form.get("date")      # dd-mm-yyyy
    time_input = request.form.get("time")      # HH:MM
    amount_input = request.form.get("amount")
    type_input = request.form.get("type")      # Requested / Debited

    # Validate
    if not all([upi_id, date_input, time_input, amount_input, type_input]):
        return jsonify({
            "error": "Missing required fields"
        }), 400

    try:

        upi_id_mod = transform_text(upi_id)

        num_char = len(upi_id_mod)

        amount = float(amount_input)

        # dd-mm-yyyy -> ddmmyyyy
        date_mod = int(date_input.replace("-", ""))

        # HH:MM -> HHMM
        time_mod = int(time_input.replace(":", ""))

        type_mod = tr_label.transform([type_input])[0]

    except Exception as e:
        return jsonify({
            "error": f"Invalid input: {str(e)}"
        }), 400

    input_df = pd.DataFrame({
        "num_char_modified": [num_char],
        "new_r_id": [upi_id_mod],
        "Amount": [amount],
        "Date_mod": [date_mod],
        "Time_mod": [time_mod],
        "Type_mod": [type_mod]
    })

    prediction = model_rfc.predict(input_df)
    probability = model_rfc.predict_proba(input_df)

    fraud = prediction[0]
    risk_score = int(probability[0][1] * 100)

    return jsonify({
        "Fraud_Result": str(fraud == 1),
        "Risk_Score": str(risk_score)
    })


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
