"""P8: why do told questions fail on Chinese real speech? Classify every told question of a zh_v3 result.

Classes (first that applies):
  hit            answered correctly
  not_routed     the question was never routed to memory (listener did not hear a question there)
  never_written  no write before the question carries the expected value under any label
  label_miss     the value was written, but under a label the question's anchor does not match (P8)
  content        label matched a write, answer still wrong (gate, stale value, ASR-damaged value)
"""
import json, re, sys, collections

def chars(s):
    return set(re.sub(r"[\s，。、？！,.?!「」：:|\[\]()（）]", "", s or ""))

def overlap(a, b):
    ca = chars(a)
    return len(ca & chars(b)) / max(1, len(ca))

def classify(path):
    d = json.load(open(path, encoding="utf-8"))
    out, rows = collections.Counter(), []
    for r in d["questions"]:
        q = r["q"]
        if q["expect"] is None:
            continue
        # writes heard before the question (same day earlier, or an earlier day; sleep writes of earlier days)
        before = [w for w in d["writes"] if w["day"] < q["day"] or (w["day"] == q["day"] and w.get("src") == "listen")]
        holders = [w for w in before if overlap(q["expect"], w["fact"][2]) >= 0.6]
        anchor = None
        if r["routed"]:
            cands = [a for a in d["answers"] if a["day"] == q["day"] and a["answer"] == r["answer"] and a["tail"] == r["tail"]]
            anchor = (cands[0]["anchor"] or "") if cands else ""
        if r.get("hit"):
            c = "hit"
        elif not r["routed"]:
            c = "not_routed"
        elif not holders:
            c = "never_written"
        else:
            ent = (anchor or "").strip("[]").split("|")[0]
            labels = sorted({w["fact"][0] for w in holders})
            c = "label_miss" if all(overlap(ent, L) < 0.5 and overlap(L, ent) < 0.5 for L in labels) else "content"
        out[c] += 1
        rows.append({"class": c, "kind": q["kind"], "q": q["text"], "entity": q["entity"], "expect": q["expect"],
                     "anchor": anchor, "answer": r["answer"][:40],
                     "written_as": sorted({"|".join(w["fact"]) for w in holders})[:3]})
    return out, rows

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for p in sys.argv[1:]:
        c, rows = classify(p)
        print("==", p.split("/")[-1], dict(c), "total", sum(c.values()))
        if "-v" in sys.argv[0:1] or True:
            for r in rows:
                if r["class"] in ("label_miss", "content", "never_written"):
                    print(f"  {r['class']:13s} {r['kind']:13s} {r['entity']}→{r['expect']} | anchor {r['anchor']} | ans {r['answer']!r} | written {r['written_as']}")
