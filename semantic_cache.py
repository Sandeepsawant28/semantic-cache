import time
import numpy as np
from embedder import Embedder


class SemanticCache:
    def __init__(self, threshold=0.90, ttl_seconds=3600, embedder=None):
        self.embedder = embedder or Embedder()
        self.threshold = threshold
        self.ttl = ttl_seconds

        # Parallel lists: index i describes one cache entry
        self.prompts = []
        self.namespaces = []
        self.responses = []
        self.expiries = []
        self.vectors = None          # matrix: one row per entry

        self.hits = 0
        self.misses = 0

    def lookup(self, prompt, namespace):
        q = self.embedder.embed(prompt)
        info = {"vector": q, "score": None, "matched_prompt": None}
        now = time.time()

        # Only consider entries with the same settings that haven't expired
        valid = [
            i for i in range(len(self.prompts))
            if self.namespaces[i] == namespace and self.expiries[i] > now
        ]
        if not valid:
            self.misses += 1
            return None, info

        # One matrix multiplication = similarity against every entry at once
        sims = self.vectors[valid] @ q
        k = int(np.argmax(sims))
        best = valid[k]

        info["score"] = float(sims[k])
        info["matched_prompt"] = self.prompts[best]

        if info["score"] >= self.threshold:
            self.hits += 1
            return self.responses[best], info

        self.misses += 1
        return None, info

    def set(self, prompt, namespace, response, vector):
        self.prompts.append(prompt)
        self.namespaces.append(namespace)
        self.responses.append(response)
        self.expiries.append(time.time() + self.ttl)

        row = vector.reshape(1, -1)
        self.vectors = row if self.vectors is None else np.vstack([self.vectors, row])

    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / total if total else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(rate, 2),
            "entries": len(self.prompts),
        }