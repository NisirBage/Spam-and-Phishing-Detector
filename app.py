"""Flask web app for the Spam & Phishing Detector.

Two independent scikit-learn pipelines (loaded from models/) score:
  - free-text email/SMS content -> spam probability
  - a URL -> phishing/malicious probability

Both routes return a 0-100% probability plus a Low/Medium/High threat
bucket, and a short list of the signals that drove the score.
"""
import pathlib

from flask import Flask, render_template, request

from src.pipelines import EmailPipeline, URLPipeline  # noqa: F401  (needed for joblib.load)
import joblib

ROOT = pathlib.Path(__file__).resolve().parent
EMAIL_MODEL_PATH = ROOT / "models" / "email_model.joblib"
URL_MODEL_PATH = ROOT / "models" / "url_model.joblib"

app = Flask(__name__)

_email_model = None
_url_model = None


def get_email_model():
    global _email_model
    if _email_model is None:
        _email_model = joblib.load(EMAIL_MODEL_PATH)
    return _email_model


def get_url_model():
    global _url_model
    if _url_model is None:
        _url_model = joblib.load(URL_MODEL_PATH)
    return _url_model


def threat_level(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/check-email", methods=["POST"])
def check_email():
    text = request.form.get("email_text", "").strip()
    result = None
    if text:
        model = get_email_model()
        prob = float(model.predict_proba([text])[0][1])
        result = {
            "input": text,
            "probability": round(prob * 100, 1),
            "level": threat_level(prob),
            "verdict": "Spam / Phishing" if prob >= 0.5 else "Likely Legitimate",
            "signals": model.top_terms(text),
        }
    return render_template("index.html", email_result=result, email_text=text)


@app.route("/check-url", methods=["POST"])
def check_url():
    url = request.form.get("url_text", "").strip()
    result = None
    if url:
        model = get_url_model()
        prob = float(model.predict_proba([url])[0][1])
        result = {
            "input": url,
            "probability": round(prob * 100, 1),
            "level": threat_level(prob),
            "verdict": "Phishing / Malicious" if prob >= 0.5 else "Likely Legitimate",
            "signals": model.top_signals(url),
        }
    return render_template("index.html", url_result=result, url_text=url)


if __name__ == "__main__":
    app.run(debug=True)
