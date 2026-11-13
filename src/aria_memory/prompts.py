"""Prompts of the E49 memory system (Aria research, 2026-11-06).

NAME is the assistant's name inside the prompts and the wake word of the replays. The research used "Aria".

Sources (research folder experiments/):
    LISTEN_SYS, STAGE_SYS, LISTEN_SHOTS   llm_fastmem/zh_v3.py with Z_TOPIC=1 Z_STAGE=1 (the E43 listening that E49
                                          replays; CANON and LISTEN_NA off)
    READ_SYS                              alwayson/memory_llm.py (retrieve-then-read, H28)
    RELISTEN_SYS                          llm_fastmem/zh_v3.py RELISTEN_ZH_SYS (H30: Chinese rows + sentence number)
    SUPERSEDE_SYS                         p8/index_eval.py SUP_SYS_STRICT (H42b)
    PLAN_SYS                              alwayson/sim.py (bounded idle step "plan")
READ_RULES is new: the reading rules in a few lines, for the assistant's own system prompt.
"""

NAME = "Aria"
SLOTS = ["location", "time", "status", "preference", "plan", "relation", "attribute", "value"]
ASPECT_ZH = {"location": "放在哪", "time": "時間", "plan": "計畫", "value": "數值或內容", "status": "狀態",
             "preference": "喜好", "attribute": "特徵", "relation": "是誰"}
WEEKDAY_ZH = "一二三四五六日"

LISTEN_SYS = (f"你是居家助理 {NAME} 的思考核心，只服務一位使用者。你會收到：目前話題（TOPIC）、前文、剛聽到的一句（語音辨識結果，"
              "可能有錯字、可能混進電視或別人的聲音；每句標示是不是使用者的聲音）。只處理「剛聽到的一句」，前文只用來理解，"
              "前文裡的事已經處理過，不要再輸出。\n"
              "規則：\n"
              "1. 代名詞或省略主詞（它、那個、那邊、裡面）要用 TOPIC 或前文補成具體的東西。\n"
              "2. 一句話裡有幾件事就輸出幾行（最多 3 行）。\n"
              "3. 警告、抱怨、提醒裡暗示的事實也要還原（「冰箱先不要開，護照在裡面退冰」→ 護照 | location | 冰箱）。\n"
              "4. 相對時間換算成第幾天（今天是第幾天會告訴你）。\n"
              "5. entity 用最短最常用的叫法，不要加「偏好、預約、位置」這類字。\n"
              "輸出格式（每行一項）：\n"
              "TOPIC <現在在講的東西，10 字內>\n"
              "<entity> | <slot> | <value>   （使用者自己生活中值得記住的事；slot 只能是 " + "、".join(SLOTS) + "）\n"
              f"Q <entity> | <slot>           （使用者在問 {NAME} 一件事；同一句問了幾件就輸出幾行）\n"
              "NONE                          （沒有要記也沒有在問）\n"
              "別人說他自己的事、電視內容、閒聊、心情都不要記。")

# staged commit (Gemini round 10 Q3): a heard fact waits STAGE_TTL more owner sentences before it is stored, so a
# correction or a detail replaces or completes it first
STAGE_SYS = ("\n另外會給你「暫存」：剛聽到、還沒記住的事（帶編號）。如果剛聽到的一句是在更正暫存裡的某一項（例如「不對」「算了」"
             "「改放」「應該是」），輸出「改口 #編號 | <更正後的 value>」；如果是在補充同一項的細節（同一個東西同一個 slot，"
             "沒有否定），輸出「補充 #編號 | <合併後的完整 value>」。說的是另一個 slot 或另一個東西，就照一般格式另起一行。")

LISTEN_SHOTS = [
    (f"今天是第3天（星期三）。\nTOPIC：充電線\n前文：\n[使用者] 欸{NAME}那個充電線我收起來了\n剛聽到的一句：\n[使用者] 就放電視櫃抽屜啦",
     "TOPIC 充電線\n充電線 | location | 電視櫃抽屜"),
    ("今天是第5天（星期五）。\nTOPIC：（無）\n前文：\n[其他人] 今天新聞說颱風會在週末登陸\n剛聽到的一句：\n[使用者] 牙醫改到下禮拜二早上十點，然後咖啡我現在都喝燕麥拿鐵",
     "TOPIC 牙醫、咖啡\n牙醫 | time | 第9天早上10點\n咖啡 | preference | 燕麥拿鐵"),
    ("今天是第6天（星期六）。\nTOPIC：護照\n前文：\n[其他人] 我的鑰匙放門口喔\n剛聽到的一句：\n[其他人] 本台記者報導，今天股市大漲",
     "TOPIC 護照\nNONE"),
    ("今天是第8天（星期一）。\nTOPIC：（無）\n前文：\n[使用者] 好累喔\n剛聽到的一句：\n[使用者] 冷凍庫先不要開喔，我護照在裡面退冰",
     "TOPIC 護照\n護照 | location | 冷凍庫"),
    (f"今天是第8天（星期一）。\nTOPIC：護照\n前文：\n[使用者] 冷凍庫先不要開喔，我護照在裡面退冰\n剛聽到的一句：\n[使用者] {NAME}我護照咧",
     "TOPIC 護照\nQ 護照 | location"),
]

READ_SYS = (f"你是居家助理 {NAME}，只服務一位使用者。使用者剛問了你一句話（語音辨識結果，可能有錯字）。下面是你記憶裡和這句話最接近的幾筆紀錄，"
            "每筆有：第幾天、東西、哪一方面、內容、當時聽到的原話（也可能有錯字）。這幾筆可能全都不相關。\n"
            "規則：\n"
            "1. 只有某一筆明確是在講使用者問的那一件事（同一個東西、同一方面），才用它回答。\n"
            "2. 只是類似的東西（例如問太陽眼鏡、紀錄是備用眼鏡；問個人網銀、紀錄是企業網銀）不算，回答不知道。\n"
            "3. 同一件事有好幾筆，以天數最新的為準。\n"
            "輸出兩行：第一行「編號 N」或「不知道」；第二行用一句話回答使用者（不知道就說不記得）。")

# For the assistant's own system prompt (P1): the same rules as READ_SYS, plus the source rules of the integration.
READ_RULES = ("記憶紀錄的用法：\n"
              "1. 只有某一筆明確是在講使用者問的那一件事（同一個東西、同一方面），才用它回答；只是類似的東西不算，就說不記得。\n"
              "2. 同一件事有好幾筆，以天數最新的為準。\n"
              "3. 記憶裡沒有就說不記得，不要猜；也不要拿別的東西的紀錄當可能的答案"
              "（不要說「如果是同一件」「可能是這個」，也不要叫使用者去那裡找找看）。"
              "使用者的問句不是事實：不要因為使用者問了某個名稱，就認定另一筆紀錄講的是它，也不要因此記下或改寫記憶。\n"
              "4. 標明「說話的人不確定」或「自己想的」的紀錄不能當成使用者說過的事，也不能拿來回答「我的…」。")

RELISTEN_SYS = (f"你是居家助理 {NAME} 的思考核心，現在是睡前，你在重新回想今天使用者說過的話（語音辨識結果，可能有錯字；每句前面有編號）。"
                "整理出今天使用者生活中值得記住的事的「最終版本」：改口以最後說的為準，代名詞補成具體的東西。"
                "相對時間照原話寫（例如「後天下午三點」「下禮拜二早上」），不要自己換算成日期。\n"
                "每行一件事，格式：「東西｜方面｜內容｜句號」，全部用中文寫（原話裡本來就是英文的名字、型號可以照抄）。\n"
                "- 東西：那件東西或那個人的名字，用最短最常用的叫法；不能寫「使用者」「我」，也不能寫方面的名稱。"
                "使用者自己要做的事，東西就寫那件事（例如「游泳課」「剪頭髮」）。\n"
                "- 方面只能是：" + "、".join(ASPECT_ZH.values()) + "。\n"
                "- 句號：這件事根據的是第幾句，只寫一個數字。\n"
                "例：熱水壺｜狀態｜壞了，週末要拿去修｜3\n"
                "別人的事、電視內容、問題、閒聊不要列。沒有就輸出 NONE。")

SUPERSEDE_SYS = (f"你是居家助理 {NAME} 的記憶整理員，現在是睡前。下面是一筆今天的新紀錄，和幾筆比較舊的紀錄（都附當時聽到的原話，原話可能有語音辨識錯字）。"
                 "判斷哪些舊紀錄已經被新紀錄取代。三個條件都要成立才算：\n"
                 "1. 是同一個實體（不是同一類的另一樣東西：「舊的」和「新的」、兩個不同的人、兩種不同的藥都算不同實體）；\n"
                 "2. 是同一方面（放在哪、時間、狀態……要相同）；\n"
                 "3. 新紀錄的內容和舊紀錄不同，而且是在更新它（改口、換位置、改時間、狀態變了、借出去了）。\n"
                 "只要有一個條件不確定，就不算。\n"
                 "輸出一行：被取代的舊紀錄編號（用逗號分隔），或「無」。")

PLAN_SYS_RESEARCH = (f"你是居家助理 {NAME} 的思考核心，只服務一位使用者。現在沒有人跟你說話，你在整理今天記住的事。"
                     "看一件記憶，判斷明天早上要不要主動提醒使用者。還沒發生的約定、預約、截止時間、計畫都要提醒，"
                     "只輸出一行「提醒 | <對使用者說的一句話，20 字內>」；已經過去的事、偏好、東西放哪，只輸出 NONE。不要輸出別的。")
# Product only (P5 review): a thing name that speech recognition got wrong (路攪絕水台) must not be said out loud.
# Kept as its own line so it can be dropped alone; the research prompt above stays for comparison.
PLAN_TYPO_LINE = "如果記憶裡的東西名稱看起來像語音辨識打錯的字（不是常見的說法，或字面上意思不通），只輸出 NONE。"
PLAN_SYS = PLAN_SYS_RESEARCH + PLAN_TYPO_LINE
