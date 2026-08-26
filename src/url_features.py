"""Lexical/host-based feature engineering for URL phishing detection.

No network calls (no DNS/WHOIS/live fetch) are made here on purpose: the
model only looks at the URL string itself, so it stays fast, works offline,
and never leaks a user-submitted URL to a third party at inference time.
"""
import math
import pathlib
import re
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_TRUSTED_DOMAINS_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "trusted_domains.txt"
try:
    TRUSTED_DOMAINS = frozenset(
        line.strip().lower() for line in _TRUSTED_DOMAINS_PATH.read_text().splitlines() if line.strip()
    )
except FileNotFoundError:
    TRUSTED_DOMAINS = frozenset()

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly", "tiny.cc", "rb.gy",
}

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "info",
    "loan", "click", "gdn", "kim", "men", "date",
}

BRAND_KEYWORDS = {
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "webscr", "password", "wallet", "billing", "invoice",
    "suspend", "unlock", "support",
}

IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _normalize(url: str) -> str:
    url = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        url = "http://" + url
    return url


class _FallbackParsed:
    scheme = ""
    netloc = ""
    path = ""
    query = ""

    @property
    def port(self):
        return None


def _safe_urlparse(url: str):
    try:
        return urlparse(url)
    except ValueError:
        return _FallbackParsed()


def _safe_port(parsed):
    try:
        return parsed.port
    except ValueError:
        return None


def extract_url_features(raw_url: str) -> dict:
    url = _normalize(raw_url)
    parsed = _safe_urlparse(url)
    raw_host = parsed.netloc.split("@")[-1].split(":")[0].lower()
    path = parsed.path or ""
    query = parsed.query or ""

    # Scheme (http/https) and a leading "www." are cosmetic conventions,
    # not phishing signal - and different URL sources are wildly
    # inconsistent about including them. Every length/count/entropy
    # feature below is computed on this canonical form so that
    # "example.com", "www.example.com" and "https://example.com" all
    # produce (almost) identical feature vectors.
    host = raw_host[4:] if raw_host.startswith("www.") else raw_host
    full = host + path + (("?" + query) if query else "")

    labels = host.split(".") if host else []
    registrable = ".".join(labels[-2:]) if len(labels) >= 2 else host
    subdomain_count = max(len(labels) - 2, 0)
    tld = labels[-1] if labels else ""

    digits = sum(c.isdigit() for c in full)

    return {
        "url_length": len(full),
        "host_length": len(host),
        "path_length": len(path),
        "num_dots": full.count("."),
        "num_hyphens": full.count("-"),
        "num_underscores": full.count("_"),
        "num_slashes": full.count("/"),
        "num_digits": digits,
        "digit_ratio": digits / len(full) if full else 0.0,
        "num_special_chars": len(re.findall(r"[@%=&?~^]", full)),
        "num_at_symbols": full.count("@"),
        "num_query_params": query.count("=") if query else 0,
        "subdomain_count": subdomain_count,
        "has_ip_host": 1 if IP_RE.match(host) else 0,
        "uses_https": 1 if parsed.scheme == "https" else 0,
        "is_shortener": 1 if registrable in SHORTENER_DOMAINS else 0,
        "suspicious_tld": 1 if tld in SUSPICIOUS_TLDS else 0,
        "has_port": 1 if _safe_port(parsed) else 0,
        "host_entropy": _shannon_entropy(host),
        "brand_keyword_count": sum(1 for k in BRAND_KEYWORDS if k in full.lower()),
        "double_slash_in_path": 1 if "//" in path else 0,
        "path_to_host_ratio": (len(path) / len(host)) if host else 0.0,
        "is_trusted_domain": 1 if registrable in TRUSTED_DOMAINS else 0,
    }


FEATURE_NAMES = list(extract_url_features("http://example.com/").keys())


def canonical_text(raw_url: str) -> str:
    """Scheme- and www-stripped host+path+query, used as the input to the
    char n-gram vectorizer so it learns from real content (brand tokens,
    suspicious words, TLDs) instead of scheme/www formatting artifacts."""
    url = _normalize(raw_url)
    parsed = _safe_urlparse(url)
    host = parsed.netloc.split("@")[-1].split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    rest = parsed.path or ""
    if parsed.query:
        rest += "?" + parsed.query
    return host + rest


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Turns a list of raw URL strings into a numeric feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [extract_url_features(u) for u in X]
        return np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
