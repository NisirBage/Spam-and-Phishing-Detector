"""Train the URL phishing/malicious-link classifier.

Dataset: ~420k labeled URLs (good/bad) aggregated from public blacklist
and benign-crawl sources (the same corpus widely used in malicious-URL
detection tutorials/papers). "bad" covers phishing, malware and spam
links; we treat it as the phishing/threat class for this project.

Only lexical/host features are used (see src/url_features.py) - no
live DNS/WHOIS/network lookups - so the model works offline and never
has to make an outbound request about a URL a user typed in.
"""
import pathlib

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.pipelines import URLPipeline
from src.url_features import TRUSTED_DOMAINS

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "url_data_raw.csv"
SAMPLE_PATH = ROOT / "data" / "url_dataset_sample.csv"
MODEL_PATH = ROOT / "models" / "url_model.joblib"

SAMPLE_SIZE_PER_CLASS = 15000
TRUSTED_REPLICAS = 8  # how many times each trusted-domain variant is repeated


def load_trusted_domain_rows() -> pd.DataFrame:
    """The base dataset happens to have zero "good" URLs without a path
    (every legitimate example was scraped with a trailing slash/path),
    so a model trained on it alone learns "no path -> phishing" and
    flags bare domains like google.com. Real browsing is full of bare
    domains, so we inject a curated list of well-known, real domains
    (the same TRUSTED_DOMAINS allow-list used as a feature at inference
    time) in their bare/https/www forms as extra "good" examples,
    replicated so it isn't washed out by the size of the base dataset.
    """
    variants = []
    for d in sorted(TRUSTED_DOMAINS):
        variants += [d, f"www.{d}", f"https://{d}", f"https://www.{d}", f"http://{d}"]
    urls = variants * TRUSTED_REPLICAS
    return pd.DataFrame({"url": urls, "label": "good"})


def load_data() -> pd.DataFrame:
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
    else:
        df = pd.read_csv(SAMPLE_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].map({"good": 0, "bad": 1})
    df = df.dropna(subset=["label"])
    return df


def make_reproducible_sample(df: pd.DataFrame):
    """Write a balanced, git-committable sample so the model can be
    retrained without needing the full 22MB raw download."""
    parts = []
    for cls in (0, 1):
        sub = df[df["label"] == cls]
        n = min(SAMPLE_SIZE_PER_CLASS, len(sub))
        parts.append(sub.sample(n=n, random_state=42))
    sample = pd.concat(parts).sample(frac=1, random_state=42)
    sample["label"] = sample["label"].map({0: "good", 1: "bad"})
    sample[["url", "label"]].to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote reproducible sample ({len(sample)} rows) to {SAMPLE_PATH}")


def main():
    df = load_data()
    if not RAW_PATH.exists():
        print(f"Note: {RAW_PATH.name} not found, trained from committed sample instead.")
    else:
        make_reproducible_sample(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["url"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    trusted = load_trusted_domain_rows()
    X_train = pd.concat([X_train, trusted["url"]], ignore_index=True)
    y_train = pd.concat([y_train, trusted["label"].map({"good": 0, "bad": 1})], ignore_index=True)
    print(f"Added {len(trusted)} trusted-domain examples to the training set.")

    pipeline = URLPipeline()
    pipeline.fit(X_train.tolist(), y_train.tolist())

    y_pred = pipeline.predict(X_test.tolist())
    print("=== URL phishing model evaluation (held-out test split) ===")
    print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing/malicious"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
