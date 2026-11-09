import hashlib
import json
import time

class ExactCache:
    def __init__(self, ttl_seconds=3600):
        self.store = {}              # key -> (response, expiry_time)
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def make_key(self, model, messages, temperature):
        # Everything that can change the answer must be in the key
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key):
        item = self.store.get(key)
        if item is None:
            self.misses += 1
            return None
        response, expiry = item
        if time.time() > expiry:     # TTL expired
            del self.store[key]
            self.misses += 1
            return None
        self.hits += 1
        return response

    def set(self, key, response):
        self.store[key] = (response, time.time() + self.ttl)

    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / total if total else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(rate, 2)}