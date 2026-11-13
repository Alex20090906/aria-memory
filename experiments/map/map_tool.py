"""Research map tool.

research_map.json is the single source for the map (nodes = goal, conjectures U, hypotheses H, experiments E,
components C, problems P, platforms I; links = [source, target, kind]).

    python map_tool.py           validate, write 研究地圖.md (the text index Claude reads to regain context)
                                 and map_doc.json (the document pushed to the star-map page's database)
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "research_map.json")
TYPES = {"goal": "目標", "conjecture": "研究者的猜想", "hypothesis": "假設", "experiment": "實驗", "component": "系統部件",
         "problem": "未解問題", "platform": "算力平台"}
STATUS = {"confirmed": "成立", "partial": "部分成立", "refuted": "不成立", "running": "進行中", "planned": "計畫中",
          "open": "未解決", "blocked": "卡住（需要研究者）", "done": "完成", "active": "運作中"}
KINDS = {"motivates": "引出", "tests": "檢驗", "supports": "支持", "refutes": "推翻", "implements": "實作",
         "part_of": "屬於", "uses": "使用", "replaces": "取代", "blocks": "卡住", "runs_on": "跑在"}


def main():
    m = json.load(open(SRC, encoding="utf-8"))
    ids = {n["id"]: n for n in m["nodes"]}
    errs = []
    if len(ids) != len(m["nodes"]):
        errs.append("duplicate node ids")
    for n in m["nodes"]:
        if n["type"] not in TYPES:
            errs.append(f"{n['id']}: unknown type {n['type']}")
        if n["status"] not in STATUS:
            errs.append(f"{n['id']}: unknown status {n['status']}")
    seen = set()
    for s, t, k in m["links"]:
        if s not in ids or t not in ids:
            errs.append(f"link {s}->{t}: unknown node")
        if k not in KINDS:
            errs.append(f"link {s}->{t}: unknown kind {k}")
        if (s, t, k) in seen:
            errs.append(f"duplicate link {s}->{t} {k}")
        seen.add((s, t, k))
    if errs:
        print("\n".join(errs))
        sys.exit(1)
    nb = defaultdict(list)
    for s, t, k in m["links"]:
        nb[s].append(f"{KINDS[k]}→{t}")
        nb[t].append(f"{s}→{KINDS[k]}")
    lines = [f"# 研究地圖（文字索引）", "", f"更新：{m['updated']}　目前焦點：{m['focus']}", "",
             "給 Claude 自己恢復脈絡用：每個節點一行，後面是它的關聯。星圖頁面讀同一份資料。", ""]
    for ty, label in TYPES.items():
        ns = [n for n in m["nodes"] if n["type"] == ty]
        if not ns:
            continue
        lines += [f"## {label}", ""]
        for n in ns:
            lines.append(f"- **{n['id']} {n['title']}**（{STATUS[n['status']]}）：{n['summary']}  ")
            if nb[n["id"]]:
                lines.append(f"  關聯：{'、'.join(nb[n['id']])}")
        lines.append("")
    open(os.path.join(HERE, "研究地圖.md"), "w", encoding="utf-8").write("\n".join(lines))
    doc = {"updated": m["updated"], "focus": m["focus"], "nodes": m["nodes"],
           "links": [{"s": s, "t": t, "k": k} for s, t, k in m["links"]]}
    raw = json.dumps(doc, ensure_ascii=False)
    open(os.path.join(HERE, "map_doc.json"), "w", encoding="utf-8").write(raw)
    print(f"ok: {len(ids)} nodes, {len(m['links'])} links, doc {len(raw.encode('utf-8')) // 1024} KiB")


if __name__ == "__main__":
    main()
