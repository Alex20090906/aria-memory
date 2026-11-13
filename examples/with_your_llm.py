"""How the pieces fit together with your own LLM.

    python examples/with_your_llm.py            # runs with a scripted stand-in for the LLM
    python examples/with_your_llm.py --openai http://localhost:8080/v1 MODEL   # an OpenAI-compatible server (untested)

The stand-in returns the replies a listener and a reader are expected to give, so the flow can be followed without a
model. The --openai path is a minimal client for any OpenAI-compatible chat endpoint (llama.cpp server, vLLM,
Ollama ...); it was not used in the research, whose thinker was an LLM server running Qwen3.8-Flash-Next.
"""
import asyncio
import json
import sys
import urllib.request

from aria_memory import CharVocab, Listener, NetIndex, read


class ScriptedLLM:
    """Replies in the formats LISTEN_SYS and READ_SYS ask for."""

    def __init__(self):
        self.replies = [
            "TOPIC 護照\n護照 | location | 書房防潮箱",           # day 1: listening
            "TOPIC 咖啡\n咖啡 | preference | 冰燕麥拿鐵",          # day 2
            "TOPIC 護照\n護照 | location | 黑色後背包夾層",        # day 3
            "編號 3\n護照在黑色後背包夾層。",                      # day 4: reading (record 3 is the newest passport)
        ]

    async def __call__(self, messages, max_tokens):
        return self.replies.pop(0)


def openai_ask(base_url: str, model: str):
    async def ask(messages, max_tokens):
        body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "User-Agent": "aria-memory"})
        def call():
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        return await asyncio.to_thread(call)
    return ask


async def main(ask):
    listener, index, records = Listener(), NetIndex(CharVocab()), {}

    def store(recs):
        for rec in recs:
            rec["id"] = len(records)
            records[rec["id"]] = rec
            index.add(rec["id"], rec)

    said = [(1, "我護照收進書房防潮箱了"), (2, "我現在喝咖啡都改冰燕麥拿鐵"), (3, "護照我改放黑色後背包夾層了")]
    for day, sentence in said:
        r = await listener.hear(ask, sentence, day=day, src="owner")
        store(r.records)              # a fact leaves staging after 2 more owner sentences (a correction would replace it)
        store(listener.flush())       # end of each day: whatever is still staged is stored
    print("records:", [(r["id"], r["thing"], r["value"]) for r in records.values()])

    question = "Aria，我護照咧？"
    top = [records[i] for i, _ in index.query(question, k=5)]
    answer = await read(ask, question, top, now_day=4)
    print("question:", question)
    print("answer:", answer["status"], answer["reply"], "| from record", answer["rec_id"], "| heard:", answer["quote"])


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--openai":
        asyncio.run(main(openai_ask(sys.argv[2], sys.argv[3])))
    else:
        asyncio.run(main(ScriptedLLM()))
