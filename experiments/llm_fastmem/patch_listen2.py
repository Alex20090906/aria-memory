import ast
import os
import sys

p = os.path.join(sys.argv[1], "zh_memory_test.py")
L = open(p, encoding="utf-8").read().split("\n")
a = next(i for i, l in enumerate(L) if l.startswith("LISTEN_SHOTS = ["))
b = next(i for i, l in enumerate(L) if l.startswith("THINK_SYS = ("))
shots = [
    ("今天是第3天（星期三）。", "[使用者] 欸 Aria 那個充電線我收起來了", "[使用者] 就放電視櫃抽屜啦", "充電線 | location | 電視櫃抽屜"),
    ("今天是第5天（星期五）。", "[其他人] 今天新聞說颱風會在週末登陸", "[使用者] 牙醫改到下禮拜二早上十點，煩欸", "牙醫 | time | 第9天早上10點"),
    ("今天是第6天（星期六）。", "[其他人] 我的鑰匙放門口喔", "[其他人] 本台記者報導，今天股市大漲", "NONE"),
    ("今天是第8天（星期一）。", "[使用者] 我護照收進書房防潮箱了", "[使用者] 我護照咧 Aria", "Q 護照 | location"),
    ("今天是第8天（星期一）。", "[使用者] 咖啡我現在都喝燕麥拿鐵", "[使用者] 鑰匙放在鞋櫃上面，不對，是鞋櫃裡面第二層", "鑰匙 | location | 鞋櫃裡面第二層"),
    ("今天是第9天（星期二）。", "[使用者] 牙醫改到明天早上十點", "[使用者] 好累喔今天加班到十點", "NONE"),
]
new = ["LISTEN_SHOTS = ["]
for d, ctx, last, ans in shots:
    user = d + "\n前文：\n" + ctx + "\n剛聽到的一句：\n" + last
    new.append(f"    ({user!r},")
    new.append(f"     {ans!r}),")
new.append("]")
L[a:b] = new
s = "\n".join(L)
s = s.replace('''    msgs.append({"role": "user", "content": f"今天是第{day}天（星期{WD[(day - 1) % 7]}）。\\n" + "\\n".join(context)})''',
              '''    msgs.append({"role": "user", "content": f"今天是第{day}天（星期{WD[(day - 1) % 7]}）。\\n前文：\\n"
                 + ("\\n".join(context[:-1]) or "（無）") + "\\n剛聽到的一句：\\n" + context[-1]})''')
# no python-side store of facts (that would be an external database): drop the dedupe dict
s = s.replace("    writes, answers, stored = [], [], {}", "    writes, answers = [], []")
s = s.replace('''                if stored.get((e, s)) == v:
                    continue                      # same fact, same value: nothing new to write
                stored[(e, s)] = v
''', "")
ast.parse(s)
open(p, "w", encoding="utf-8").write(s)
print("ok", "剛聽到的一句：\\n\" + context[-1]" in s, "stored" in s)
