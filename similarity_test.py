from embedder import Embedder, cosine_similarity

emb = Embedder()

# ---------- Part 1: similarity table ----------
sentences = [
    "What is the capital of France?",
    "Tell me France's capital city.",
    "Which city is the capital of France?",
    "What is the population of France?",
    "How does photosynthesis work?",
    "How do I reset my password?",
    "I forgot my password, how can I change it?",
]

vectors = emb.embed_many(sentences)

print("=== Similarity table ===")
for i, a in enumerate(sentences):
    print(f"\n{a}")
    for j, b in enumerate(sentences):
        if i != j:
            score = cosine_similarity(vectors[i], vectors[j])
            print(f"   {score:.2f}  {b}")

# ---------- Part 2: tricky pairs ----------
# (sentence A, sentence B, do they really mean the same thing?)
tricky_pairs = [
    ("What is the capital of France?", "Tell me France's capital city.", True),
    ("How do I reset my password?", "I forgot my password, how can I change it?", True),
    ("hello", "hi there", True),
    ("What is the capital of France?", "What is the capital of Germany?", False),
    ("Python is easy to learn", "Python is hard to learn", False),
    ("Book a flight to Paris", "Cancel my flight to Paris", False),
    ("What is the capital of France?", "What is the population of France?", False),
    ("How are you?", "What's the weather today?", False),
]

print("\n\n=== Tricky pairs ===")
print(f"{'score':>6}  {'same?':<6}  pair")
for a, b, same in tricky_pairs:
    score = cosine_similarity(emb.embed(a), emb.embed(b))
    print(f"{score:>6.2f}  {str(same):<6}  {a}  <->  {b}")

# ---------- Part 3: can one threshold separate them? ----------
same_scores = [cosine_similarity(emb.embed(a), emb.embed(b))
               for a, b, same in tricky_pairs if same]
diff_scores = [cosine_similarity(emb.embed(a), emb.embed(b))
               for a, b, same in tricky_pairs if not same]

print("\n=== Threshold check ===")
print(f"lowest score among SAME-meaning pairs:      {min(same_scores):.2f}")
print(f"highest score among DIFFERENT-meaning pairs: {max(diff_scores):.2f}")
if min(same_scores) > max(diff_scores):
    print("A threshold between these two numbers separates them cleanly.")
else:
    print("Overlap! No single threshold is perfect. You must pick which error to accept.")
    