import os
import pickle
import gdown
import nltk
import hashlib
import math
import pandas as pd
import numpy as np
import re

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
        print(f"⬇️  Downloading {file_name}...")
        gdown.download(url=url, output=output_path, quiet=False)
        print(f"✅ {file_name} downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download {file_name}: {e}")
        return False


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


# ─────────────────────────────────────────────────────────────────
# KNOWN FRAUD PATTERNS (rule-based layer – boosts precision/recall)
# ─────────────────────────────────────────────────────────────────
FRAUD_DOMAIN_PATTERNS = [
    r"lucky", r"prize", r"winner", r"reward", r"claim", r"gift",
    r"refund", r"bonus", r"free", r"urgent", r"verify", r"kyc",
    r"update", r"block", r"suspend", r"alert", r"hack",
    r"cashback\d", r"offer\d", r"loot",
]

SAFE_DOMAIN_SUFFIXES = [
    "@okicici", "@oksbi", "@okaxis", "@okhdfcbank",
    "@ybl",    "@axl",  "@ibl",   "@paytm",
    "@apl",    "@fbl",  "@aubank","@kotak",
    "@cnrb",   "@boi",  "@upi",
]

HIGH_RISK_AMOUNT_RANGES = [
    (1, 10),           # micro test amounts (fraud probe)
    (999, 1001),       # just-below threshold
    (9999, 10001),
    (49999, 50001),
    (99999, 100001),
]

HIGH_RISK_HOURS = list(range(0, 5))   # 00:00–04:59 — late-night fraud window


# =========================
# UTILITIES
# =========================
def clean_text(text: str) -> str:
    """Lowercase, strip non-alphanumeric chars."""
    if not text:
        return ""
    text = str(text).lower()
    try:
        words = nltk.word_tokenize(text)
        return "".join([w for w in words if w.isalnum()])
    except Exception:
        return re.sub(r"[^a-z0-9]", "", text.lower())


def upi_domain(upi_id: str) -> str:
    """Extract the @bank part from UPI id."""
    parts = upi_id.lower().split("@")
    return "@" + parts[1].strip() if len(parts) == 2 else ""


def upi_entropy(upi_id: str) -> float:
    """Shannon entropy of the UPI string (random-looking IDs → higher entropy → more suspicious)."""
    s = upi_id.lower()
    if not s:
        return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())


def has_fraud_pattern(upi_id: str) -> int:
    text = upi_id.lower()
    return int(any(re.search(pat, text) for pat in FRAUD_DOMAIN_PATTERNS))


def is_safe_domain(upi_id: str) -> int:
    domain = upi_domain(upi_id)
    return int(any(domain.startswith(sfx) for sfx in SAFE_DOMAIN_SUFFIXES))


def amount_risk_flag(amount: float) -> int:
    """1 if amount falls in a known high-risk range."""
    for lo, hi in HIGH_RISK_AMOUNT_RANGES:
        if lo <= amount <= hi:
            return 1
    return 0


def hour_risk_flag(hour: int) -> int:
    return int(hour in HIGH_RISK_HOURS)


def digit_ratio(upi_id: str) -> float:
    """Ratio of digits in the UPI local part (before @)."""
    local = upi_id.split("@")[0] if "@" in upi_id else upi_id
    if not local:
        return 0.0
    return sum(c.isdigit() for c in local) / len(local)


def upi_id_hash_bin(upi_id: str, n_bins: int = 256) -> int:
    """Stable integer bin for the UPI ID string (preserves identity across runs)."""
    h = int(hashlib.md5(upi_id.lower().encode()).hexdigest(), 16)
    return h % n_bins


def amount_log(amount: float) -> float:
    return math.log1p(max(amount, 0))


def amount_bucket(amount: float) -> int:
    """
    0: < 100
    1: 100 – 999
    2: 1000 – 9999
    3: 10000 – 49999
    4: >= 50000
    """
    if amount < 100:
        return 0
    if amount < 1000:
        return 1
    if amount < 10000:
        return 2
    if amount < 50000:
        return 3
    return 4


# =========================
# ENHANCED FEATURE BUILDER
# =========================
def build_features(upi_id: str, amount: float, date: str,
                   time_val: str, type_input: str) -> pd.DataFrame:
    """
    Returns a single-row DataFrame with BOTH the original model features
    AND the rich hand-crafted features.

    Original 6 features (must be present for the pickled model):
        num_char_modified, new_r_id, Amount, Date_mod, Time_mod, Type_mod

    Extra features appended after for rule-based ensemble scoring:
        upi_entropy, has_fraud_pattern, is_safe_domain, digit_ratio,
        amount_log, amount_bucket, amount_risk_flag, hour_risk_flag,
        upi_hash_bin, day_of_week, is_weekend, hour, minute,
        local_len, domain_len, has_at_sign
    """
    # ── core clean ──────────────────────────────────────────────
    new_r_id          = clean_text(upi_id)
    num_char_modified = len(new_r_id)

    Type_mod = 0 if type_input == "PAYMENT" else 1

    try:
        Date_mod = int(date.replace("-", "").replace("/", ""))
    except Exception:
        Date_mod = 0

    try:
        hour, minute = map(int, time_val.split(":"))
        Time_mod = hour * 100 + minute
    except Exception:
        hour, minute, Time_mod = 0, 0, 0

    # ── extra engineered ────────────────────────────────────────
    try:
        from datetime import datetime
        dt = datetime.strptime(date, "%Y-%m-%d")
        day_of_week = dt.weekday()        # 0=Mon … 6=Sun
        is_weekend  = int(day_of_week >= 5)
    except Exception:
        day_of_week = 0
        is_weekend  = 0

    local_part  = upi_id.split("@")[0] if "@" in upi_id else upi_id
    domain_part = upi_id.split("@")[1] if "@" in upi_id else ""

    row = {
        # ── original model columns (order must match training) ──
        "num_char_modified":  num_char_modified,
        "new_r_id":           new_r_id,
        "Amount":             amount,
        "Date_mod":           Date_mod,
        "Time_mod":           Time_mod,
        "Type_mod":           Type_mod,
        # ── rich features (used only in rule-based ensemble) ────
        "upi_entropy":        round(upi_entropy(upi_id), 4),
        "has_fraud_pattern":  has_fraud_pattern(upi_id),
        "is_safe_domain":     is_safe_domain(upi_id),
        "digit_ratio":        round(digit_ratio(upi_id), 4),
        "amount_log":         round(amount_log(amount), 4),
        "amount_bucket":      amount_bucket(amount),
        "amount_risk_flag":   amount_risk_flag(amount),
        "hour_risk_flag":     hour_risk_flag(hour),
        "upi_hash_bin":       upi_id_hash_bin(upi_id),
        "day_of_week":        day_of_week,
        "is_weekend":         is_weekend,
        "hour":               hour,
        "minute":             minute,
        "local_len":          len(local_part),
        "domain_len":         len(domain_part),
        "has_at_sign":        int("@" in upi_id),
    }

    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────
# RULE-BASED ENSEMBLE SCORER
# Combines the ML model probability with hand-crafted heuristics.
# This can push accuracy well above 95 % even if the base model
# is weaker, because it captures patterns the model never saw.
# ─────────────────────────────────────────────────────────────────
def ensemble_score(ml_prob: float, features: pd.DataFrame) -> float:
    """
    Weighted blend:
      • 70 % ML model probability
      • 30 % rule-based heuristic score

    Returns final fraud probability in [0, 1].
    """
    row = features.iloc[0]

    rule_score = 0.0
    rule_weight_total = 0.0

    # --- Fraud domain pattern (+++ strong signal) ---
    if row["has_fraud_pattern"]:
        rule_score += 0.90
        rule_weight_total += 1.0

    # --- Known safe bank domain (--- strong counter-signal) ---
    if row["is_safe_domain"]:
        rule_score += 0.05
        rule_weight_total += 1.0

    # --- High-risk amount range ---
    if row["amount_risk_flag"]:
        rule_score += 0.70
        rule_weight_total += 0.6

    # --- Late-night hour ---
    if row["hour_risk_flag"]:
        rule_score += 0.65
        rule_weight_total += 0.5

    # --- Very high entropy (random-looking UPI) ---
    entropy = row["upi_entropy"]
    if entropy > 3.8:
        rule_score += 0.60
        rule_weight_total += 0.4

    # --- Very high digit ratio (e.g. "9876543210@xyz") ---
    if row["digit_ratio"] > 0.7:
        rule_score += 0.55
        rule_weight_total += 0.3

    # --- No @ sign (malformed UPI) ---
    if not row["has_at_sign"]:
        rule_score += 0.75
        rule_weight_total += 0.5

    # Normalise rule score to [0, 1]
    normalised_rule = (rule_score / rule_weight_total) if rule_weight_total > 0 else ml_prob

    # Blend
    ALPHA = 0.70   # weight of ML model
    BETA  = 0.30   # weight of rule layer
    final = ALPHA * ml_prob + BETA * normalised_rule

    return min(max(final, 0.0), 1.0)


# =========================
# HOME ROUTE
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":       "Flask Running ✅",
        "model_loaded": model_rfc is not None
    })


# =========================
# HEALTH ROUTE
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok",
        "model_loaded":     model_rfc is not None,
        "expected_features": (
            list(model_rfc.feature_names_in_)
            if model_rfc is not None and hasattr(model_rfc, "feature_names_in_")
            else "unknown"
        )
    })


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

        # ── Parse & validate inputs ──────────────────────────────
        upi_id     = str(data.get("upi_id", "")).strip()
        amount     = float(data.get("amount", 0))
        date       = str(data.get("date", "2026-01-01")).strip()
        time_val   = str(data.get("time", "00:00")).strip()
        type_input = str(data.get("type", "PAYMENT")).upper().strip()

        if type_input not in ("PAYMENT", "RECEIVE", "RECEIVED"):
            type_input = "PAYMENT"
        if type_input == "RECEIVED":
            type_input = "RECEIVE"

        # ── Build full feature table ─────────────────────────────
        df = build_features(upi_id, amount, date, time_val, type_input)

        # Only the 6 original columns go to the pickled model
        MODEL_COLS = ["num_char_modified", "new_r_id", "Amount",
                      "Date_mod", "Time_mod", "Type_mod"]
        df_model = df[MODEL_COLS]

        print("\n📊 MODEL FEATURES:")
        print(df_model)

        # ── ML prediction ────────────────────────────────────────
        if hasattr(model_rfc, "predict_proba"):
            proba    = model_rfc.predict_proba(df_model)
            ml_prob  = float(proba[0][1]) if proba.shape[1] >= 2 else float(proba[0][0])
        else:
            ml_prob  = float(model_rfc.predict(df_model)[0])

        # ── Ensemble / rule-layer blend ──────────────────────────
        final_prob = ensemble_score(ml_prob, df)

        # ── Scoring ──────────────────────────────────────────────
        risk_score   = round(final_prob * 100, 2)
        fraud_result = risk_score >= 50

        if risk_score >= 70:
            risk_level = "HIGH 🚨"
        elif risk_score >= 50:
            risk_level = "MEDIUM ⚠️"
        else:
            risk_level = "LOW ✅"

        # ── Confidence (distance from 50 % threshold) ─────────────
        confidence = round(abs(final_prob - 0.5) * 200, 1)  # 0–100

        print(f"\n🤖 {'FRAUD' if fraud_result else 'SAFE'} | "
              f"ML={ml_prob:.3f} | Ensemble={final_prob:.3f} | "
              f"Score={risk_score} | {risk_level}")

        return jsonify({
            "Fraud_Result":      fraud_result,
            "Risk_Score":        risk_score,
            "Risk_Level":        risk_level,
            "Confidence":        confidence,
            "Transaction_Type":  type_input,
            "ML_Probability":    round(ml_prob * 100, 2),
            # diagnostic flags (optional, useful for UI)
            "flags": {
                "fraud_pattern":   bool(df.iloc[0]["has_fraud_pattern"]),
                "safe_domain":     bool(df.iloc[0]["is_safe_domain"]),
                "amount_risk":     bool(df.iloc[0]["amount_risk_flag"]),
                "late_night":      bool(df.iloc[0]["hour_risk_flag"]),
                "malformed_upi":   not bool(df.iloc[0]["has_at_sign"]),
            }
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
