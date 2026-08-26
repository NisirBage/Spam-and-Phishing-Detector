import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.url_features import extract_url_features, canonical_text
from src.email_features import extract_email_features


def test_www_and_scheme_are_normalized_away():
    a = extract_url_features("example.com")
    b = extract_url_features("www.example.com")
    c = extract_url_features("http://www.example.com")
    for key in ("host_length", "url_length", "num_dots", "subdomain_count"):
        assert a[key] == b[key] == c[key], key


def test_trusted_domain_is_recognized_with_or_without_www():
    assert extract_url_features("google.com")["is_trusted_domain"] == 1
    assert extract_url_features("www.google.com")["is_trusted_domain"] == 1
    assert extract_url_features("totally-not-google.com")["is_trusted_domain"] == 0


def test_ip_host_is_detected():
    assert extract_url_features("http://192.168.1.1/login")["has_ip_host"] == 1
    assert extract_url_features("http://example.com/login")["has_ip_host"] == 0


def test_suspicious_tld_and_shortener_flags():
    assert extract_url_features("http://scam.tk/verify")["suspicious_tld"] == 1
    assert extract_url_features("http://bit.ly/abc123")["is_shortener"] == 1


def test_brand_keyword_count_matches_suspicious_words():
    feats = extract_url_features("http://example.com/login/verify/account")
    assert feats["brand_keyword_count"] >= 3


def test_malformed_url_does_not_raise():
    # IPv6-looking garbage historically crashed urlparse; must degrade
    # gracefully instead of raising.
    extract_url_features("http://[::1")


def test_canonical_text_strips_scheme_and_www():
    assert canonical_text("https://www.example.com/path") == "example.com/path"
    assert canonical_text("example.com/path") == "example.com/path"


def test_email_features_flag_urgency_and_money_words():
    spammy = extract_email_features("URGENT!! You have WON a FREE prize, claim now!!!")
    normal = extract_email_features("Hey, are we still on for lunch tomorrow?")
    assert spammy["has_urgency_words"] == 1
    assert spammy["has_money_words"] == 1
    assert spammy["num_exclamations"] > normal["num_exclamations"]


def test_email_features_handle_empty_string():
    feats = extract_email_features("")
    assert feats["length"] == 0
    assert feats["word_count"] == 0
