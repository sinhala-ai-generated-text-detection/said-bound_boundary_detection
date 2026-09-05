"""Detectors, from trivial baselines up to a linear text model.

The baselines are not filler. On this dataset the AI class is a minority, and
boundary positions are deliberately sampled in a band rather than uniformly, so
two things can inflate a score without any language understanding at all:
predicting the majority class, and predicting from sentence position alone.
`PositionPrior` measures exactly that second effect, which makes it the number
every text model has to beat before its score means anything.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------- baselines ---

class AllHuman:
    name = "baseline: all-human"

    def fit(self, docs):
        return self

    def predict(self, docs):
        return [0] * sum(d.n for d in docs)


class AllAI:
    name = "baseline: all-AI"

    def fit(self, docs):
        return self

    def predict(self, docs):
        return [1] * sum(d.n for d in docs)


class RandomPrior:
    """Coin flip weighted by the training AI rate. Seeded."""
    name = "baseline: random (train prior)"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.p = 0.5

    def fit(self, docs):
        tot = sum(d.n for d in docs)
        ai = sum(sum(d.labels) for d in docs)
        self.p = ai / tot if tot else 0.5
        return self

    def predict(self, docs):
        rng = np.random.default_rng(self.seed)
        n = sum(d.n for d in docs)
        return (rng.random(n) < self.p).astype(int).tolist()


class PositionPrior:
    """Predicts from normalised sentence position only - no text at all.

    Bins position in the document and predicts the majority label seen in that
    bin during training. Quantifies how much of any score is positional
    artefact rather than real detection.
    """
    name = "baseline: position only (no text)"

    def __init__(self, bins: int = 10):
        self.bins = bins
        self.table = np.zeros(bins)

    @staticmethod
    def _pos(si: int, n: int) -> float:
        return si / max(1, n - 1)

    def _bin(self, si: int, n: int) -> int:
        return min(self.bins - 1, int(self._pos(si, n) * self.bins))

    def fit(self, docs):
        num = np.zeros(self.bins)
        den = np.zeros(self.bins)
        for d in docs:
            for si, y in enumerate(d.labels):
                b = self._bin(si, d.n)
                num[b] += y
                den[b] += 1
        self.table = np.divide(num, den, out=np.zeros(self.bins), where=den > 0)
        return self

    def predict(self, docs):
        out = []
        for d in docs:
            for si in range(d.n):
                out.append(int(self.table[self._bin(si, d.n)] >= 0.5))
        return out


# ---------------------------------------------------------- linear model ---

class LinearDetector:
    """Char n-gram TF-IDF over each sentence, logistic regression per sentence.

    Character n-grams rather than words because Sinhala is highly inflected and
    agglutinative, so char n-grams capture morphology and orthographic habit
    without depending on a word tokenizer.

    Optionally concatenates the neighbouring sentences as a second feature
    block, so the model can see local discontinuity instead of judging each
    sentence in isolation, which is what boundary detection depends on.
    """

    def __init__(self, *, ngram=(2, 4), min_df=2, max_features=200_000,
                 C=1.0, context=0, use_position=False, seed=42,
                 feat_map=None, text=True):
        self.ngram = ngram
        self.min_df = min_df
        self.max_features = max_features
        self.C = C
        self.context = context
        self.use_position = use_position
        self.seed = seed
        # Optional masked-LM likelihood features, keyed by record_id.
        self.feat_map = feat_map or {}
        # text=False gives a likelihood-only detector: it reads no characters
        # at all, which isolates how much of the task the LM signal alone
        # solves.
        self.text = text
        self.feat_dim = (len(next(iter(self.feat_map.values()))[0])
                         if self.feat_map else 0)
        self._mu = None
        self._sd = None
        self.vec = None
        self.ctx_vec = None
        self.clf = None

    @property
    def name(self) -> str:
        if not self.text:
            return f"likelihood only (C={self.C})"
        bits = [f"char{self.ngram[0]}-{self.ngram[1]}", f"C={self.C}"]
        if self.context:
            bits.append(f"ctx={self.context}")
        if self.use_position:
            bits.append("pos")
        if self.feat_map:
            bits.append("likelihood")
        return "linear (" + ", ".join(bits) + ")"

    # -- feature construction ------------------------------------------
    @staticmethod
    def _flat(docs):
        S, P = [], []
        for d in docs:
            for si, s in enumerate(d.sentences):
                S.append(s)
                P.append(si / max(1, d.n - 1))
        return S, P

    def _context_strings(self, docs):
        """For each sentence, its neighbours joined: the local background."""
        out = []
        k = self.context
        for d in docs:
            for si in range(d.n):
                lo = max(0, si - k)
                hi = min(d.n, si + k + 1)
                out.append(" ".join(d.sentences[j] for j in range(lo, hi)
                                    if j != si))
        return out

    def _likelihood_block(self, docs, fit: bool):
        rows = []
        for d in docs:
            f = self.feat_map.get(d.record_id)
            for i in range(d.n):
                rows.append(list(f[i]) if f and i < len(f)
                            else [0.0] * self.feat_dim)
        arr = np.asarray(rows, dtype=float)
        if fit:
            self._mu = arr.mean(0)
            sd = arr.std(0)
            sd[sd < 1e-6] = 1.0
            self._sd = sd
        return csr_matrix((arr - self._mu) / self._sd)

    def _matrix(self, docs, fit: bool):
        S, P = self._flat(docs)
        if not self.text:
            blocks = [self._likelihood_block(docs, fit)]
            if self.use_position:
                blocks.append(csr_matrix(np.array(P).reshape(-1, 1)))
            return hstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]
        if fit:
            self.vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=self.ngram,
                min_df=self.min_df, max_features=self.max_features,
                sublinear_tf=True)
            X = self.vec.fit_transform(S)
        else:
            X = self.vec.transform(S)
        blocks = [X]

        if self.context:
            Cs = self._context_strings(docs)
            if fit:
                self.ctx_vec = TfidfVectorizer(
                    analyzer="char_wb", ngram_range=self.ngram,
                    min_df=self.min_df, max_features=self.max_features,
                    sublinear_tf=True)
                Xc = self.ctx_vec.fit_transform(Cs)
            else:
                Xc = self.ctx_vec.transform(Cs)
            blocks.append(Xc)

        if self.feat_map:
            blocks.append(self._likelihood_block(docs, fit))

        if self.use_position:
            blocks.append(csr_matrix(np.array(P).reshape(-1, 1)))

        return hstack(blocks).tocsr() if len(blocks) > 1 else X

    # -- api ------------------------------------------------------------
    def fit(self, docs):
        X = self._matrix(docs, fit=True)
        y = [v for d in docs for v in d.labels]
        self.clf = LogisticRegression(
            C=self.C, max_iter=2000, class_weight="balanced",
            random_state=self.seed)
        self.clf.fit(X, y)
        return self

    def predict(self, docs):
        return self.clf.predict(self._matrix(docs, fit=False)).astype(int).tolist()

    def predict_proba(self, docs):
        return self.clf.predict_proba(self._matrix(docs, fit=False))[:, 1]


# ------------------------------------------------------------- smoothing ---

def smooth_runs(docs, flat_pred, min_run: int = 2):
    """Remove implausibly short author runs.

    Every construction in this dataset produces contiguous AI spans of at least
    one sentence and usually two, so isolated single-sentence flips are far more
    often noise than a real authorship change. Applied as a post-process so its
    effect stays visible as a separate row in the results table.
    """
    out: list[int] = []
    i = 0
    for d in docs:
        seq = list(flat_pred[i:i + d.n])
        i += d.n
        j = 0
        while j < len(seq):
            k = j
            while k + 1 < len(seq) and seq[k + 1] == seq[j]:
                k += 1
            run = k - j + 1
            if run < min_run and j > 0 and k < len(seq) - 1:
                seq[j:k + 1] = [seq[j - 1]] * run
            j = k + 1
        out.extend(seq)
    return out
