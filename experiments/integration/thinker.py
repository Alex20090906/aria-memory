"""Model 3, the thinker: LLM server (OpenAI-compatible API) with one tool, `recall`, served by the
memory model. The thinker decides what to try to remember, like a person searching memory.

API key: read from the LLM_API_KEY environment variable, never stored in files.
"""
import json
import os
import time
import urllib.request

BASE = os.environ.get("LLM_BASE_URL", "http://<thinker-host>:8081/v1")
MODEL = "qwen3.8-flash-next-iq3_s"

SYSTEM = (
    "You are Aria's thinking mind. Aria's awareness hands you a request it heard. You have no notes "
    "or database; your only access to the past is your own memory through the tool `recall`. "
    "Memories are stored as short facts like \"Mia's keys is in the green bowl.\" To remember, give "
    "the BEGINNING of such a fact as the cue, e.g. \"Mia's keys is\", and memory completes it. "
    "The result has: recalled (the completion), differs_from_guess (true if memory changed what you "
    "would have guessed anyway; false means the completion is probably just a guess), "
    "recalled_familiarity (0-1, how strongly this exact fact feels experienced; above 0.4 is a real "
    "memory, below 0.15 means it is probably not what happened). "
    "If the detail feels unfamiliar or is only a guess, say honestly that you do not remember the "
    "detail; never invent it. Answer in one or two short sentences."
)

TOOLS = [{"type": "function", "function": {
    "name": "recall",
    "description": "Try to remember: memory completes the beginning of a fact.",
    "parameters": {"type": "object", "properties": {
        "cue": {"type": "string", "description": "Beginning of the fact, e.g. \"Omar's wallet is\""}},
        "required": ["cue"]}}}]


def _post(body, timeout=180):
    key = os.environ["LLM_API_KEY"].strip()
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def think(request, recall_fn=None, max_rounds=4, log=None):
    """Run the thinker on one request. recall_fn(cue) -> dict, or None for the no-memory baseline."""
    msgs = [{"role": "system", "content": SYSTEM if recall_fn else SYSTEM.split(" You have no notes")[0]
             + " Answer in one or two short sentences. If you do not know, say so."},
            {"role": "user", "content": request}]
    trace, t0 = [], time.time()
    for _ in range(max_rounds):
        body = {"model": MODEL, "messages": msgs, "max_tokens": 1024, "temperature": 0.2}
        if recall_fn:
            body["tools"] = TOOLS
        r = _post(body)["choices"][0]["message"]
        calls = r.get("tool_calls") or []
        msgs.append({k: v for k, v in r.items() if k in ("role", "content", "tool_calls")})
        if not calls:
            return {"answer": (r.get("content") or "").strip(), "trace": trace, "seconds": round(time.time() - t0, 1)}
        for c in calls:
            try:
                cue = json.loads(c["function"]["arguments"]).get("cue", "")
            except Exception:
                cue = ""
            res = recall_fn(cue) if cue else {"error": "empty cue"}
            trace.append(res)
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps(res)})
    return {"answer": "(no answer)", "trace": trace, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    fake = lambda cue: {"cue": cue, "recalled": "in the blue box", "differs_from_guess": True,
                        "recalled_familiarity": 0.8}
    print(think("[MIC] Leo: Aria, where did I put my passport?", fake))
