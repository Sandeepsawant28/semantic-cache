import os
import time
from dotenv import load_dotenv
from groq import Groq
from cache_v1 import ExactCache

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
cache = ExactCache(ttl_seconds=600)

def cached_chat(prompt, model="openai/gpt-oss-20b", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    key = cache.make_key(model, messages, temperature)

    cached = cache.get(key)
    if cached is not None:
        return cached, "HIT"

    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    answer = resp.choices[0].message.content
    cache.set(key, answer)
    return answer, "MISS"

if __name__ == "__main__":
    prompts = [
        "What is the capital of France?",
        "What is the capital of France?",
        "Tell me France's capital city.",
    ]
    for p in prompts:
        start = time.time()
        answer, status = cached_chat(p)
        print(f"[{status}] {time.time()-start:.2f}s  {p}")
    print(cache.stats())