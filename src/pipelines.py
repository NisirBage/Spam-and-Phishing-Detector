"""Model pipeline classes.

Kept in their own always-imported-as-a-module file (never executed as
__main__) so joblib pickles a stable `src.pipelines.EmailPipeline` /
`src.pipelines.URLPipeline` reference that both the training scripts
and app.py can unpickle.
"""
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.email_features import EmailFeatureExtractor
from src.url_features import URLFeatureExtractor, FEATURE_NAMES, canonical_text


class EmailPipeline:
    """TF-IDF + engineered-feature union -> Logistic Regression."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            min_df=2, max_features=8000,
        )
        self.feature_extractor = EmailFeatureExtractor()
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)

    def fit(self, texts, y):
        X_tfidf = self.tfidf.fit_transform(texts)
        X_num = self.scaler.fit_transform(self.feature_extractor.transform(texts))
        X = hstack([X_tfidf, X_num]).tocsr()
        self.clf.fit(X, y)
        return self

    def _transform(self, texts):
        X_tfidf = self.tfidf.transform(texts)
        X_num = self.scaler.transform(self.feature_extractor.transform(texts))
        return hstack([X_tfidf, X_num]).tocsr()

    def predict(self, texts):
        return self.clf.predict(self._transform(texts))

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._transform(texts))

    def top_terms(self, text: str, top_n: int = 6):
        """TF-IDF terms present in `text` with the strongest spam-leaning
        coefficients, for a human-readable explanation."""
        vec = self.tfidf.transform([text])
        vocab = self.tfidf.get_feature_names_out()
        coefs = self.clf.coef_[0][: len(vocab)]
        idx = vec.nonzero()[1]
        scored = sorted(
            ((vocab[i], coefs[i]) for i in idx), key=lambda t: t[1], reverse=True
        )
        return [term for term, score in scored[:top_n] if score > 0]


class URLPipeline:
    """Character n-grams of the raw URL + engineered lexical/host
    features -> Logistic Regression.

    The char n-grams (e.g. "pay-pal", "-verify", ".tk/") let the model
    pick up brand-impersonation and suspicious-token patterns directly
    from the string, which also protects against a structural feature
    (like "has a path") accidentally becoming a shortcut for the label
    on any one training set.
    """

    def __init__(self):
        self.extractor = URLFeatureExtractor()
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=6000,
        )
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)

    def _numeric(self, urls, fit=False):
        feats = self.extractor.transform(urls)
        return self.scaler.fit_transform(feats) if fit else self.scaler.transform(feats)

    def fit(self, urls, y):
        X_char = self.tfidf.fit_transform([canonical_text(u) for u in urls])
        X_num = self._numeric(urls, fit=True)
        X = hstack([X_char, X_num]).tocsr()
        self.clf.fit(X, y)
        return self

    def _transform(self, urls):
        X_char = self.tfidf.transform([canonical_text(u) for u in urls])
        X_num = self._numeric(urls)
        return hstack([X_char, X_num]).tocsr()

    def predict(self, urls):
        return self.clf.predict(self._transform(urls))

    def predict_proba(self, urls):
        return self.clf.predict_proba(self._transform(urls))

    def top_signals(self, url: str, top_n: int = 5):
        """Engineered features that most influenced this specific
        prediction (scaled value x coefficient, engineered block only)."""
        feats = self.extractor.transform([url])
        scaled = self.scaler.transform(feats)[0]
        n_char = len(self.tfidf.get_feature_names_out())
        coefs = self.clf.coef_[0][n_char:]
        contrib = sorted(
            zip(FEATURE_NAMES, feats[0], scaled * coefs),
            key=lambda t: t[2],
            reverse=True,
        )
        return [(name, val) for name, val, score in contrib[:top_n] if score > 0]
