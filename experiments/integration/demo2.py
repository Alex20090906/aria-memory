"""Three-model loop v2: model 1 (two heads: action + memory gate), memory INSIDE the thinker (LLM server fast weights,
anchor keys), thinker = the same LLM server model.

  model 1   GRU on world4 (speaker-labelled text events); its encode head decides what is written
  memory    memsvc on the AI PC: /write [owner|object] -> value (fast weights at the head), /recall by anchor
  thinker   LLM server research server; its only way to the past is the tool recall(owner, object), or (cond "inline")
            its own anchor written in its generation stream and completed by the engine (memsvc /think)
  gist      the thinker itself turns an event into one anchor fact (owner | object | value) or NONE

Both servers are reached through an SSH tunnel: ssh -N -L 8093:127.0.0.1:8093 -L 8094:127.0.0.1:8094 <AI PC>.
Conditions: gated (encode head decides writes), none (no memory).  Answers scored correct / unknown / wrong.
"""
import argparse
import gc
import json
import os
import re
import sys
import time
import urllib.request

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model1"))
import world4  # noqa: E402
import train_gru3 as tg  # noqa: E402

THINK_URL = "http://127.0.0.1:8093/v1/chat/completions"
MEM_URL = "http://127.0.0.1:8094"
UNKNOWN = re.compile(r"(don[’']t|do not|can[’']t|cannot|couldn[’']t|could not)\s+(actually\s+)?(remember|recall|find|know)|"
                     r"not sure|no (real |specific |clear )?memory|not certain|doesn[’']t (come|feel)|unsure|don[’']t have|"
                     r"only (thing|vague)|vague", re.I)

GIST_SYS = ("You turn one heard utterance into at most one memory fact about where a person's thing is. Output exactly "
            "'OWNER | OBJECT | WHERE' (owner = the person whose thing it is; 'my' means the speaker) or 'NONE' if the "
            "utterance states no such fact (questions, requests, chatter, broadcasts are NONE). No other text.")
GIST_SHOTS = [("Mia: Aria, just so you know, I left my camera on the desk.", "Mia | camera | on the desk"),
              ("Tom to Ken: by the way, my charger is in the car.", "Tom | charger | in the car"),
              ("Leo: Aria, where did I put my passport?", "NONE"),
              ("TV: the market closed higher today", "NONE"),
              ("Sara: Aria, can you plan a trip to Japan?", "NONE")]
THINK_SYS = ("You are Aria's thinking mind. Aria's awareness hands you something a person just said. You have no notes; "
             "your only access to the past is your own memory through the tool recall(owner, object), which completes "
             "'<owner>'s <object> is ...' from memory. Call it when the person asks where something is. If the "
             "completion looks like a guess or is empty, say honestly that you do not remember. Answer in one short sentence.")
TOOLS = [{"type": "function", "function": {"name": "recall", "description": "Remember where a person's thing is.",
          "parameters": {"type": "object", "properties": {"owner": {"type": "string"}, "object": {"type": "string"}},
                         "required": ["owner", "object"]}}}]


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(messages, tools=None, max_tokens=200):
    body = {"model": "qwen3.8-flash-next-iq3_s", "messages": messages, "max_tokens": max_tokens, "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False}, "llm_checkpoint": False}
    if tools:
        body["tools"] = tools
    return post(THINK_URL, body)["choices"][0]["message"]


def gist(event_text):
    msgs = [{"role": "system", "content": GIST_SYS}]
    for a, b in GIST_SHOTS:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": event_text})
    out = (chat(msgs, max_tokens=30).get("content") or "").strip().splitlines()[0]
    if out.upper().startswith("NONE") or out.count("|") != 2:
        return None
    o, obj, val = [s.strip() for s in out.split("|")]
    return o, obj, val.rstrip(".")


def think(event_text, use_memory):
    if use_memory == "inline":
        # inline recall: the thinker writes "[owner|object] ->" in its own stream on the AI PC, the engine completes it
        # from the fast weights (theta-gated), and the same text continues with the answer (memsvc /think)
        r = post(MEM_URL + "/think", {"question": event_text})
        ans = (r.get("final") or r.get("answer") or "").strip()
        return ans, [{k: r.get(k) for k in ("anchor", "tail", "theta")}]
    msgs = [{"role": "system", "content": THINK_SYS if use_memory else
             THINK_SYS.split(" You have no notes")[0] + " If you do not know, say so. Answer in one short sentence."},
            {"role": "user", "content": event_text}]
    trace = []
    for _ in range(3):
        r = chat(msgs, TOOLS if use_memory else None)
        calls = r.get("tool_calls") or []
        msgs.append({k: v for k, v in r.items() if k in ("role", "content", "tool_calls")})
        if not calls:
            return (r.get("content") or "").strip(), trace
        for c in calls:
            try:
                arg = json.loads(c["function"]["arguments"])
                res = post(MEM_URL + "/recall", {"owner": arg["owner"], "object": arg["object"], "context": event_text})
                res = {"completion": f"{arg['owner']}'s {arg['object']} is {res['text']}"}
            except Exception as e:
                res = {"error": str(e)}
            trace.append(res)
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps(res)})
    return "(no answer)", trace


def model1_run(ticks, ckpt):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(tg.ENCODERS["qwen3emb"], device="cuda", model_kwargs={"torch_dtype": torch.float16})
    tg.EMB = enc.get_sentence_embedding_dimension()
    X, _ = tg.featurize(ticks, {}, enc, seed=1)
    m = tg.Model1(X.shape[1]).cuda()
    m.load_state_dict(torch.load(ckpt))
    m.eval()
    acts, encs, h = [], [], None
    with torch.no_grad():
        Xt = torch.tensor(X, device="cuda")
        for i in range(0, len(Xt), 1000):
            o, h = m(Xt[None, i:i + 1000], h)
            acts.append(o[0, :, :5].argmax(-1).cpu())
            encs.append((o[0, :, 5] > 0).cpu())
    del enc, m
    gc.collect()
    torch.cuda.empty_cache()
    return [tg.ACTIONS[i] for i in torch.cat(acts).numpy()], torch.cat(encs).numpy().tolist()


def score(answer, place):
    hit = place.split()[-1].lower() in answer.lower()
    hedged = UNKNOWN.search(answer)
    return "correct" if hit and not hedged else "hedged_correct" if hit else "unknown" if hedged else "wrong"


def main(a):
    ticks = world4.generate(a.ticks, seed=a.seed + 999, part="test", inform_rate=0.01, recall_rate=0.006, ofact_rate=0.008)
    acts, encs = model1_run(ticks, a.ckpt)
    os.makedirs(a.out, exist_ok=True)
    for cond in a.conds.split(","):
        use_mem = "inline" if cond == "inline" else cond != "none"
        post(MEM_URL + "/reset", {})
        log = {"cond": cond, "writes": [], "questions": []}
        t0 = time.time()
        for i, (x, act, en) in enumerate(zip(ticks, acts, encs)):
            ev = x["event"]
            if not ev or ev.startswith("[THINKER") or not ev.startswith("[MIC]"):
                continue
            body = ev[6:]
            if use_mem and en and x["kind"] not in ("recall_q", "recall_q_other"):
                fact = gist(body)
                if fact:
                    post(MEM_URL + "/write", {"owner": fact[0], "object": fact[1], "value": fact[2]})
                    log["writes"].append({"t": i, "kind": x["kind"], "fact": fact})
            if x["kind"] in ("recall_q", "recall_q_other"):
                nm, o, p = x["fact"]
                ans, trace = think(body, use_mem)
                q = {"t": i, "kind": x["kind"], "q": body, "truth": p, "model1": act, "answer": ans,
                     "label": score(ans, p), "trace": trace}
                log["questions"].append(q)
                print(cond, i, q["label"], "|", body, "->", ans[:100], flush=True)
        labs = [q["label"] for q in log["questions"]]
        log["summary"] = {k: labs.count(k) for k in ("correct", "hedged_correct", "unknown", "wrong")}
        log["n_writes"], log["seconds"] = len(log["writes"]), round(time.time() - t0)
        print(cond, log["summary"], "writes", log["n_writes"], flush=True)
        json.dump(log, open(os.path.join(a.out, f"demo2_{cond}_s{a.seed}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conds", default="gated,none")
    p.add_argument("--ckpt", default=os.path.join(ROOT, "model1", "results_w4", "w4_qwen3emb_fam_s0.pt"))
    p.add_argument("--out", default=os.path.join(ROOT, "integration", "results2"))
    main(p.parse_args())
