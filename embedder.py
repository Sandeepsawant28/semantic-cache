import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Loads a small pretrained model (downloads ~90 MB on first run)
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        # One text -> one vector of 384 numbers, scaled to length 1
        return self.model.encode(text, normalize_embeddings=True)

    def embed_many(self, texts):
        # Many texts at once -> faster than calling embed() in a loop
        return self.model.encode(texts, normalize_embeddings=True)


def cosine_similarity(a, b):
    # For unit-length vectors, the dot product equals cosine similarity
    return float(np.dot(a, b))