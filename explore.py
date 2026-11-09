from embedder import Embedder

emb = Embedder()

v = emb.embed("What is the capital of France?")

print("type:", type(v))
print("shape:", v.shape)                                  # (384,)
print("first 5 numbers:", v[:5])
print("length (norm):", float((v ** 2).sum() ** 0.5))     # should be ~1.0