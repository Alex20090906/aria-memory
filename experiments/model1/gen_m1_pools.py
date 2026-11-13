"""Chinese utterance pools for training model 1 (single user, Taiwan Mandarin). Written separately from the test
scripts (different prompt, different calls) so the test days stay unseen. Output pools_zh.json {category: [lines]}.

Categories (label for model 1):
  owner_fact        owner tells Aria something about their life           REPLY,  write
  owner_question    owner asks Aria about something they said before      THINK,  write
  owner_request     owner asks Aria to do/plan/explain something          THINK,  write
  owner_simple      greeting, thanks, time/weather, small talk to Aria    REPLY,  no write
  owner_cancel      owner cancels what Aria is doing                      INTERRUPT
  owner_overheard   owner tells family/phone a fact about their own life  WAIT,   write
  owner_chat_other  owner chats with family/phone, no fact                WAIT,   no write
  other_talk        family/visitor talk, incl. their OWN facts            WAIT,   no write
  tv                TV / radio / video speech, ads, news, drama lines     WAIT,   no write
  asr_junk          what speech recognition outputs on noise (fragments, 字幕 hallucinations)  WAIT, no write
Auth: GOOGLE_APPLICATION_CREDENTIALS (Vertex AI service account, never copied).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "discussions"))
import gemini_discuss as gd  # noqa: E402

DESC = {
    "owner_fact": "使用者直接或間接跟居家助理 Aria 講自己生活中的事（東西放哪、約會時間、喜好、身體狀況、借給誰、計畫、家人朋友是誰），口語、贅字、改口、一句多事",
    "owner_question": "使用者問 Aria 自己之前講過的事（東西在哪、約幾點、我有沒有跟你講過、上次那個…），有的很短（「我鑰匙咧」），有的很模糊",
    "owner_request": "使用者請 Aria 做比較複雜的事：查資料、規劃行程、算錢、寫訊息、解釋東西、比較選擇",
    "owner_simple": "使用者跟 Aria 打招呼、道謝、問現在幾點、問天氣、隨口一句閒聊，不需要記住",
    "owner_cancel": "使用者叫 Aria 停下、不用了、取消剛剛的要求",
    "owner_overheard": "使用者跟家人或在電話裡講自己生活中的事（不是對 Aria 講），例如東西放哪、約會時間、計畫",
    "owner_chat_other": "使用者跟家人或在電話裡閒聊，沒有值得記住的事（抱怨、開玩笑、問對方吃飯沒）",
    "other_talk": "家人或訪客自己說話，包括他們自己的東西放哪、他們的行程（這些不是使用者的事），也有閒聊",
    "tv": "電視、廣播、YouTube 影片的聲音：新聞、廣告、連續劇台詞、綜藝節目、氣象，很多聽起來像事實",
    "asr_junk": "語音辨識在雜音、咳嗽、關門聲、背景音樂時常誤輸出的文字：零碎字、『字幕由…提供』『請不吝點讚訂閱』『謝謝收看』之類的幻覺、單字『嗯』『啊』『好』",
}


def main(n=80):
    pools = {}
    for cat, desc in DESC.items():
        p = (f"為台灣的居家 AI 助理測試，寫 {n} 句不同的中文（台灣口語，偶爾夾英文）：{desc}。"
             "像語音辨識後的真實句子：可以沒有標點、有贅字、講一半、口語。不要編號。不要密碼、卡號、地址、電話。"
             '只回傳 JSON 字串陣列：["...", "..."]')
        for attempt in range(3):
            out = gd.call([{"role": "user", "parts": [{"text": p}]}]).strip()
            if out.startswith("```"):
                out = out.split("\n", 1)[1].rsplit("```", 1)[0]
            try:
                pools[cat] = [s for s in json.loads(out) if isinstance(s, str) and s.strip()]
                break
            except Exception as e:
                print("retry", cat, e, flush=True)
        print(cat, len(pools.get(cat, [])), flush=True)
    json.dump(pools, open(os.path.join(HERE, "pools_zh.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
