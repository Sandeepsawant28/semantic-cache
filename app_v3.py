import os
import time
from dotenv import load_dotenv
from groq import Groq
from semantic_cache import SemanticCache

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
cache = SemanticCache(threshold=0.90, ttl_seconds=600)


def make_namespace(model, temperature):
    return f"{model}|{temperature}"


def cached_chat(prompt, model="openai/gpt-oss-20b", temperature=0):
    namespace = make_namespace(model, temperature)

    cached, info = cache.lookup(prompt, namespace)
    if cached is not None:
        return cached, "HIT", info

    messages = [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    answer = resp.choices[0].message.content
    cache.set(prompt, namespace, answer, info["vector"])
    return answer, "MISS", info


if __name__ == "__main__":
    prompts = [
        "What is the capital of France?",
        "What is the capital of France?",       # exact repeat
        "Tell me France's capital city.",       # paraphrase: V1 missed this
        "Which city is the capital of France?", # paraphrase
        "How does photosynthesis work?",        # unrelated: should MISS
    ]
    for p in prompts:
        start = time.time()
        answer, status, info = cached_chat(p)
        score = f"{info['score']:.2f}" if info["score"] is not None else " n/a"
        print(f"[{status:<4}] {time.time()-start:.2f}s  score={score}  {p}")
        if status == "HIT":
            print(f"        matched: {info['matched_prompt']}")
    print(cache.stats())