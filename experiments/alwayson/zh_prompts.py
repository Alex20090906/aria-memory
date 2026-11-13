"""聆聽步驟的提示與解析，照抄 llm_fastmem/zh_memory_test.py（只在快取裡沒有的句子才用本機替身聽）。"""
import re

SLOTS = ["location", "time", "status", "preference", "plan", "relation", "attribute", "value"]
LISTEN_SYS = ("你是居家助理 Aria 的思考核心，只服務一位使用者。你會收到「前文」和「剛聽到的一句」（語音辨識結果，可能有錯字、"
              "可能混進電視或別人的聲音），每句前面標示是不是使用者的聲音。只處理「剛聽到的一句」；前文只用來理解代名詞和改口，"
              "前文裡的事情已經處理過了，絕對不要再輸出前文的事實。\n"
              "entity 用最短、最常用的叫法（「咖啡」不要「咖啡偏好」，「牙醫」不要「洗牙預約」，「護照」不要「我的護照」）。\n"
              "如果最後一句是使用者說的、而且說出或暗示了使用者自己生活中值得記住的事，輸出 1 到 3 行："
              "「entity | slot | value」。entity 是簡短中文名詞；slot 只能是 " + "、".join(SLOTS) + "；value 是簡短中文片語，"
              "相對時間要換算成第幾天（例如「第17天下午3點」）。用前文解決「那邊」「它」「不對，是…」這類說法，記改口後的版本。\n"
              "如果最後一句是使用者在問 Aria 某件事，輸出一行「Q entity | slot」。\n"
              "其他情況（電視、別人說他自己的事、閒聊、心情、不是使用者的聲音）輸出 NONE。不要輸出別的文字。")
LISTEN_SHOTS = [
    ('今天是第3天（星期三）。\n前文：\n[使用者] 欸 Aria 那個充電線我收起來了\n剛聽到的一句：\n[使用者] 就放電視櫃抽屜啦',
     '充電線 | location | 電視櫃抽屜'),
    ('今天是第5天（星期五）。\n前文：\n[其他人] 今天新聞說颱風會在週末登陸\n剛聽到的一句：\n[使用者] 牙醫改到下禮拜二早上十點，煩欸',
     '牙醫 | time | 第9天早上10點'),
    ('今天是第6天（星期六）。\n前文：\n[其他人] 我的鑰匙放門口喔\n剛聽到的一句：\n[其他人] 本台記者報導，今天股市大漲',
     'NONE'),
    ('今天是第8天（星期一）。\n前文：\n[使用者] 我護照收進書房防潮箱了\n剛聽到的一句：\n[使用者] 我護照咧 Aria',
     'Q 護照 | location'),
]


def parse_listen(lines):
    facts, qs = [], []
    for line in lines:
        line = line.strip()
        if not line or line.upper().startswith("NONE"):
            continue
        if line.startswith("Q") and line.count("|") == 1:
            e, s = [x.strip() for x in line[1:].split("|")]
            if s in SLOTS and e:
                qs.append((e, s))
        elif line.count("|") == 2:
            e, s, v = [x.strip() for x in line.split("|")]
            if s in SLOTS and e and v:
                facts.append((e, s, v))
    return facts[:3], qs[:3]
