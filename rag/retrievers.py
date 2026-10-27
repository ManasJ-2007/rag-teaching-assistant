"""Retrievers: BM25 (implemented from scratch), dense embeddings, hybrid fusion and cross-encoder re-ranking.

All retrievers expose search(query, k) -> [(chunk_index, score)], best first.
"""
import math
from collections import Counter

import numpy as np

from .text import tokenize


class BM25:
    """Okapi BM25: idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len/avg_len))."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg = self.lens.mean()
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        s = np.zeros(len(self.docs))
        for term in set(tokenize(query)):
            if term not in self.idf:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
            s += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens / self.avg))
        return s

    def search(self, query, k=5):
        s = self.scores(query)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]


class Dense:
    """Cosine similarity between sentence embeddings (sentence-transformers)."""

    QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}

    def __init__(self, texts, model_name="sentence-transformers/all-MiniLM-L6-v2", embeddings=None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.emb = embeddings if embeddings is not None else self.encode(texts)

    def encode(self, texts, query=False):
        prefix = self.QUERY_PREFIX.get(self.model_name, "") if query else ""
        return self.model.encode([prefix + t for t in texts], normalize_embeddings=True,
                                 batch_size=32, show_progress_bar=False)

    def search(self, query, k=5):
        s = self.emb @ self.encode([query], query=True)[0]
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]


class Hybrid:
    """Reciprocal rank fusion: score = sum over retrievers of 1 / (60 + rank)."""

    def __init__(self, *retrievers, depth=50, c=60):
        self.retrievers, self.depth, self.c = retrievers, depth, c

    def search(self, query, k=5):
        fused = Counter()
        for r in self.retrievers:
            for rank, (i, _) in enumerate(r.search(query, self.depth), start=1):
                fused[i] += 1 / (self.c + rank)
        return fused.most_common(k)


class Reranked:
    """Retrieve `depth` candidates, then re-score each (query, chunk) pair with a cross-encoder."""

    def __init__(self, base, texts, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", depth=20):
        from sentence_transformers import CrossEncoder
        self.base, self.texts, self.depth = base, texts, depth
        self.model = CrossEncoder(model_name)

    def search(self, query, k=5):
        cand = [i for i, _ in self.base.search(query, self.depth)]
        s = self.model.predict([(query, self.texts[i]) for i in cand], show_progress_bar=False)
        order = np.argsort(-s)[:k]
        return [(cand[j], float(s[j])) for j in order]
