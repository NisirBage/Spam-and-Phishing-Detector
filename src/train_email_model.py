"""Train the email/SMS spam classifier.

Dataset: the classic UCI SMS Spam Collection (5,572 labeled messages,
ham/spam). Pipeline: TF-IDF (word 1-2 grams) + a handful of engineered
surface features, fed into a class-weight-balanced Logistic Regression
so we get calibrated-ish predict_proba scores for the threat meter.
"""
import pathlib

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.pipelines import EmailPipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sms_spam_raw.csv"
MODEL_PATH = ROOT / "models" / "email_model.joblib"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]
    df = df.dropna(subset=["text"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = EmailPipeline()
    pipeline.fit(X_train.tolist(), y_train.tolist())

    y_pred = pipeline.predict(X_test.tolist())
    print("=== Email/SMS spam model evaluation ===")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
