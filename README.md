# Spam & Phishing Detector

A small Flask app with two independent machine-learning classifiers:

1. **Email / SMS spam detector** — TF-IDF + engineered text features → Logistic Regression.
2. **URL phishing detector** — character n-grams + engineered lexical/host features → Logistic Regression.

Both return a 0–100% threat probability, a Low/Medium/High bucket, and the
top signals that drove the score. Everything runs locally: no message or
URL you submit is sent to a third party, and no live DNS/WHOIS lookups are
made — classification is based purely on the text/URL string itself.

## Why two separate models

Spam email/SMS and phishing URLs are different problems. Email spam is
mostly a **bag-of-words / language** problem (word choice, urgency,
money-related terms). Phishing URLs are mostly a **string-structure**
problem (brand impersonation, suspicious TLDs, IP-address hosts, lookalike
domains) where the "text" is short, adversarial, and often has no natural
language at all. Using one model for both would force a lossy shared
representation, so this project keeps them as two pipelines with two
different feature sets.

## Datasets

| Model | Dataset | Size | Source |
|---|---|---|---|
| Email/SMS | UCI SMS Spam Collection (ham/spam) | 5,572 messages | [mirror used](https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv), originally from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| URL | Malicious URL corpus (good/bad), aggregated from blacklist + benign-crawl sources | ~420k URLs | [faizann24/Using-machine-learning-to-detect-malicious-URLs](https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs) |

`data/sms_spam_raw.csv` (the full email dataset, ~500KB) is committed.
The full URL dataset is 22MB, so only a **reproducible balanced sample**
(`data/url_dataset_sample.csv`, 30k rows) is committed; `src/train_url_model.py`
regenerates it from the raw file when present, and falls back to the
committed sample otherwise. See [Reproducing training](#reproducing-training).

`data/trusted_domains.txt` is a small hand-curated allow-list of ~150
well-known real domains (google.com, github.com, irs.gov, …), used both as
a feature (`is_trusted_domain`) and as extra "legitimate" training
examples — see [A real bug I found while building this](#a-real-bug-i-found-while-building-this).

## Architecture

```
src/
  email_features.py   engineered numeric features for email/SMS text
  url_features.py      engineered lexical/host features for URLs
  pipelines.py          EmailPipeline / URLPipeline model classes
  train_email_model.py  trains + evaluates the email model, saves models/email_model.joblib
  train_url_model.py    trains + evaluates the URL model, saves models/url_model.joblib
app.py                  Flask routes, loads both saved models
templates/, static/     UI
```

**Email pipeline**: TF-IDF (word 1–2 grams, English stopwords removed) is
combined with a handful of engineered features (exclamation count,
uppercase-word ratio, presence of URLs/money words/urgency words) via
`scipy.sparse.hstack`, then fed to a class-weight-balanced
`LogisticRegression`.

**URL pipeline**: character n-grams (3–5, `char_wb`) of a canonicalized
form of the URL (scheme and `www.` stripped, since those are formatting
conventions, not phishing signal) are combined with ~22 engineered
lexical/host features (URL length, digit ratio, hyphen count, subdomain
count, IP-address host, suspicious TLD, URL-shortener domain, brand
keyword count, domain entropy, `is_trusted_domain`, …), then fed to a
class-weight-balanced `LogisticRegression`.

Both `top_terms()` / `top_signals()` methods expose *why* a given input
scored the way it did, which the UI shows as "contributing terms/signals".

## Results (held-out 20% test split)

**Email/SMS spam** — accuracy 0.99

| | precision | recall | f1 |
|---|---|---|---|
| ham | 0.99 | 0.99 | 0.99 |
| spam | 0.96 | 0.93 | 0.95 |

**URL phishing/malicious** — accuracy 0.96

| | precision | recall | f1 |
|---|---|---|---|
| legitimate | 0.99 | 0.97 | 0.98 |
| phishing/malicious | 0.86 | 0.96 | 0.90 |

(URL precision on the phishing class is lower than recall by design —
for a threat detector, catching 96% of malicious URLs at the cost of some
false positives is the right trade-off.)

## A real bug I found while building this

The raw URL dataset turned out to have a strong labeling artifact: **100%**
of the "good" URLs in it include a path (e.g. `example.com/page`) and
**0%** of them have a `www.` prefix or explicit `https://` scheme —
purely because of how that dataset was collected, not because those are
real phishing signals. A model trained on it directly learned "no path"
and "has www." as shortcuts for "malicious", and confidently flagged
`google.com` and `www.google.com` as ~90-100% phishing.

Fixed by:
1. Canonicalizing every URL (strip scheme + `www.`) before computing any
   feature, so formatting conventions can't leak into the model.
2. Adding character n-grams of the canonicalized URL, so the model has
   real lexical signal (brand names, suspicious words) instead of relying
   only on structural counts.
3. Injecting a curated list of well-known real domains (`data/trusted_domains.txt`)
   as extra "good" training examples in bare/`www.`/`https://` forms, plus
   an explicit `is_trusted_domain` feature, so exact matches to known-safe
   domains aren't drowned out by brand-impersonation patterns elsewhere in
   the corpus (e.g. `paypal.com` itself vs. `paypal-secure-login.tk`).

`tests/test_features.py` has regression tests for this
(`test_www_and_scheme_are_normalized_away`, `test_trusted_domain_is_recognized_with_or_without_www`).

## Known limitations

- The URL model only sees the URL string — no page content, no live
  DNS/SSL/WHOIS checks — so it can't catch a phishing page hosted on an
  otherwise-benign or freshly-registered domain, and can't verify a
  certificate is actually valid.
- A bare domain with no scheme and no path (e.g. typing `google.com` with
  nothing else) still lands in the "Medium" bucket rather than "Low",
  because the base training corpus has almost no genuine bare-domain
  examples to learn from beyond the curated allow-list.
- `trusted_domains.txt` is a small, manually curated list (~150 domains) —
  it measurably improves the specific failure mode above but isn't a
  general substitute for a real domain-reputation service.

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate  # or `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

## Reproducing training

```bash
pip install -r requirements-dev.txt
python -m src.train_email_model
python -m src.train_url_model    # uses the committed data/url_dataset_sample.csv
                                   # unless data/url_data_raw.csv is present
pytest tests/
```

To retrain on the full 420k-row URL dataset, download it into
`data/url_data_raw.csv` from the source linked in [Datasets](#datasets)
before running `train_url_model.py`.

## Stack

Python, Flask, scikit-learn, pandas, scipy, joblib.
