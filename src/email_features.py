"""Lightweight engineered signals for spam/phishing email & SMS text.

These are combined with a TF-IDF bag-of-words representation in the
training pipeline; on their own they capture surface-level "shoutiness"
and urgency cues that raw word counts under-weight.
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://|www\.", re.I)
MONEY_RE = re.compile(r"[$£€]|\bfree\b|\bwin\b|\bwon\b|\bprize\b|\bcash\b|\bclaim\b", re.I)
URGENCY_RE = re.compile(r"\burgent\b|\bimmediately\b|\bexpire[sd]?\b|\bact now\b|\bverify\b|\bsuspend", re.I)
PHONE_RE = re.compile(r"\b\d{5,}\b")


def extract_email_features(text: str) -> dict:
    text = text or ""
    words = text.split()
    upper_words = [w for w in words if len(w) > 1 and w.isupper()]
    digits = sum(c.isdigit() for c in text)

    return {
        "length": len(text),
        "word_count": len(words),
        "num_exclamations": text.count("!"),
        "num_digits": digits,
        "digit_ratio": digits / len(text) if text else 0.0,
        "uppercase_word_ratio": len(upper_words) / len(words) if words else 0.0,
        "has_url": 1 if URL_RE.search(text) else 0,
        "has_money_words": 1 if MONEY_RE.search(text) else 0,
        "has_urgency_words": 1 if URGENCY_RE.search(text) else 0,
        "has_long_number": 1 if PHONE_RE.search(text) else 0,
    }


FEATURE_NAMES = list(extract_email_features("sample text").keys())


class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Turns a list of raw message strings into a numeric feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [extract_email_features(t) for t in X]
        return np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
