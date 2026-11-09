from embedder import Embedder
from cache_v1 import ExactCache
from semantic_cache import SemanticCache

MODEL, TEMP = "fake-model", 0

# (prompt, intent). Same intent = same real meaning.
stream = [
    # 7 originals
    ("What is the capital of France?", "france_capital"),
    ("How does photosynthesis work?", "photosynthesis"),
    ("What is a Python list comprehension?", "list_comp"),
    ("Explain what an API is.", "api"),
    ("Who wrote Romeo and Juliet?", "romeo"),
    ("Python is easy to learn", "python_easy"),
    ("Book a flight to Paris", "book_flight"),
    # 5 exact repeats
    ("What is the capital of France?", "france_capital"),
    ("How does photosynthesis work?", "photosynthesis"),
    ("What is a Python list comprehension?", "list_comp"),
    ("Explain what an API is.", "api"),
    ("Who wrote Romeo and Juliet?", "romeo"),
    # 5 paraphrases (same meaning, new wording)
    ("Tell me France's capital city.", "france_capital"),
    ("Can you explain how plants do photosynthesis?", "photosynthesis"),
    ("What are list comprehensions in Python?", "list_comp"),
    ("What does API mean and what does it do?", "api"),
    ("Which author created Romeo and Juliet?", "romeo"),
    # 4 traps (look similar, but DIFFERENT meaning)
    ("Python is hard to learn", "python_hard"),
    ("Cancel my flight to Paris", "cancel_flight"),
    ("What is the capital of Germany?", "germany_capital"),
    ("What is the population of France?", "france_population"),
]

# The fake LLM "answers" with the intent label, so a hit is correct
# only if the stored answer's label matches the new prompt's intent.


def run_exact():
    cache = ExactCache(ttl_seconds=3600)
    calls = hits = wrong = 0
    for prompt, intent in stream:
        key = cache.make_key(MODEL, [{"role": "user", "content": prompt}], TEMP)
        resp = cache.get(key)
        if resp is None:
            calls += 1
            cache.set(key, intent)
        else:
            hits += 1
            wrong += resp != intent
    return calls, hits, wrong


def run_semantic(embedder, threshold):
    cache = SemanticCache(threshold=threshold, ttl_seconds=3600, embedder=embedder)
    ns = f"{MODEL}|{TEMP}"
    calls = hits = wrong = 0
    for prompt, intent in stream:
        resp, info = cache.lookup(prompt, ns)
        if resp is None:
            calls += 1
            cache.set(prompt, ns, intent, info["vector"])
        else:
            hits += 1
            wrong += resp != intent
    return calls, hits, wrong


embedder = Embedder()
ideal_calls = len({intent for _, intent in stream})

print(f"Total prompts: {len(stream)}")
print(f"A perfect cache would make {ideal_calls} API calls and have 0 wrong hits.\n")
print(f"{'cache':<18}{'API calls':>10}{'hits':>6}{'WRONG hits':>12}")

calls, hits, wrong = run_exact()
print(f"{'V1 exact':<18}{calls:>10}{hits:>6}{int(wrong):>12}")

for t in [0.6, 0.7, 0.8, 0.9, 0.95]:
    calls, hits, wrong = run_semantic(embedder, t)
    print(f"{'V3 thr=' + str(t):<18}{calls:>10}{hits:>6}{int(wrong):>12}")