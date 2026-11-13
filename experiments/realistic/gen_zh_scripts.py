"""Messy Taiwanese-Mandarin household scripts with ground truth (test level 2: synthetic speech).

One owner lives with the assistant Aria (only the owner uses it). Others are around: a family member, a visitor,
the TV, the owner on the phone. Each day is a list of segments; the owner's facts are spread the way people talk:
several facts in one sentence, one fact over two sentences, corrections ("不對，是另一個抽屜"), pronouns across
sentences ("放那邊"), relative time ("下禮拜三"), mixed Chinese-English, fillers, half sentences, talking to someone
else while Aria overhears, TV speech that sounds like facts but is not the owner's.

Output zh_scripts.json:
  days: [{"day": int, "weekday": str, "segments": [{"speaker": "owner|family|visitor|tv|phone_owner",
          "to": "aria|family|visitor|phone|none", "text": str}]}]
  facts: [{"day", "entity", "slot", "value", "evidence": "short quote", "update_of": idx|null,
           "kind": "direct|indirect|multi|split|correction|pronoun|relative_time|overheard"}]
  questions: [{"day", "text" (owner speech, casual), "entity", "slot", "expect": value valid then | null,
               "stale": old value | null, "kind": "casual|terse|vague|meta|never|similar_but_different"}]
Auth: GOOGLE_APPLICATION_CREDENTIALS (Vertex AI service account, never copied).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "discussions"))
import gemini_discuss as gd  # noqa: E402

SLOTS = ["location", "time", "status", "preference", "plan", "relation", "attribute", "value"]

PROMPT = """你要幫一個居家 AI 助理（名字叫 Aria）產生測試資料。這個助理只有一個使用者，住在台灣，講台灣口語中文，偶爾夾英文。
家裡還有一位家人、偶爾有訪客、電視常開著，使用者有時候在講電話（只聽得到使用者這邊）。

請寫第 {d0} 天到第 {d1} 天的生活片段（每天 6–12 段話）。要求像真實錄音轉出來的文字：
- 口語、贅字（欸、那個、就是、齁、啦、喔）、話講一半、改口、重複、吞字，偶爾英文單字（例如 meeting、deadline、Uber）。
- 使用者提到的「值得記住的事」要散落、自然，不要每句都是。種類要混合：
  direct（直接跟 Aria 說）、indirect（暗示，例如「冷凍庫先不要開，我護照在裡面退冰」）、
  multi（一句講兩三件事）、split（一件事拆在兩句，例如「我把那個收起來了」「就充電線啊，放電視櫃抽屜」）、
  correction（講錯再改口）、pronoun（用「那邊」「它」指前面講過的東西）、
  relative_time（「下禮拜三」「後天早上」，要能換算成第幾天）、
  overheard（使用者跟家人或電話講的，不是對 Aria 說，但 Aria 聽得到）。
- 也要有看起來像事實、但不是使用者的事：電視節目內容、家人講他自己的東西、訪客的事。這些不能被記成使用者的。
- 不要有任何密碼、卡號、身分證、地址、電話。

記憶格式：每個事實是 [entity|slot] -> value，slot 只能是 {slots}。entity 用簡短中文名詞（例如「護照」「牙醫」「媽媽」「咖啡」），
value 是簡短的中文片語（例如「冷凍庫」「禮拜五下午三點」「燕麥奶不加糖」）。

這一批要有：約 {n_facts} 個使用者的事實（其中約 {n_upd} 個更新前面的事實，用 update_of 指出被更新的事實在「這一批 facts 陣列」裡的索引），
約 {n_q} 個使用者在後面幾天問 Aria 的問題（口語、有的很短「我護照咧」、有的很模糊「我上次跟你講的那個東西放哪」、有的問「我有沒有跟你說過…」），
其中 {n_never} 個是從來沒講過的事（expect 為 null），{n_similar} 個是「講過類似但不同的事」（例如講過護照、問的是駕照；expect 為 null）。
問題的 expect 要是「問的那天」有效的值；如果之前被更新過，stale 填舊值。

只回傳 JSON：
{{"days": [{{"day": int, "weekday": "一~日", "segments": [{{"speaker": "owner|family|visitor|tv|phone_owner", "to": "aria|family|visitor|phone|none", "text": str}}]}}],
 "facts": [{{"day": int, "entity": str, "slot": str, "value": str, "evidence": str, "update_of": int|null, "kind": str}}],
 "questions": [{{"day": int, "text": str, "entity": str, "slot": str, "expect": str|null, "stale": str|null, "kind": str}}]}}
問題也要出現在 days 的 segments 裡（speaker 為 owner、to 為 aria）。"""


def ask(d0, d1):
    p = PROMPT.format(d0=d0, d1=d1, slots="、".join(SLOTS), n_facts=30, n_upd=6, n_q=24, n_never=4, n_similar=3)
    for attempt in range(3):
        out = gd.call([{"role": "user", "parts": [{"text": p}]}]).strip()
        if out.startswith("```"):
            out = out.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            return json.loads(out)
        except Exception as e:
            print("retry", e, flush=True)
    raise SystemExit("no JSON")


def main():
    allb = {"days": [], "facts": [], "questions": []}
    for d0, d1 in [(1, 7), (8, 14), (15, 21)]:
        b = ask(d0, d1)
        off = len(allb["facts"])
        for f_ in b["facts"]:
            if f_.get("update_of") is not None:
                f_["update_of"] += off
        allb["days"] += b["days"]; allb["facts"] += b["facts"]; allb["questions"] += b["questions"]
        print(d0, d1, "days", len(b["days"]), "facts", len(b["facts"]), "questions", len(b["questions"]), flush=True)
    bad = [f_ for f_ in allb["facts"] if f_["slot"] not in SLOTS]
    print("total facts", len(allb["facts"]), "questions", len(allb["questions"]), "segments",
          sum(len(d["segments"]) for d in allb["days"]), "bad slots", len(bad))
    json.dump(allb, open(os.path.join(HERE, "zh_scripts.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
