import os
H = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(H, "colloquial_test.py"); s = open(p, encoding="utf-8").read()
old = '''             "not remember. The words after the arrow are your own memory: trust them and use them.")'''
new = '''             "not remember. The words after the arrow are your own memory: trust them and use them.\n\n"
             "Examples of how you start:\n"
             'Ken says: "ugh where did I leave my charger again??" -> you write: [Ken|charger] ->\n'
             'Mia says: "Aria, my wallet?" -> you write: [Mia|wallet] ->\n'
             'Tom says: "did I ever tell you where my ring is?" -> you write: [Tom|ring] ->\n'
             'Sara says: "any idea where Leo put his scarf? he is freaking out" -> you write: [Leo|scarf] ->\n'
             "Always start your reply with the anchor line.")'''
assert old in s; s = s.replace(old, new)
old2 = '''            q = f"{speaker}: {x[kind]}"'''
new2 = '''            q = f'{speaker} says: "{x[kind]}"'   # "Name: text" made the thinker end its turn at once'''
assert old2 in s; s = s.replace(old2, new2)
s = s.replace('colloquial_result.json', 'colloquial_result_v2.json')
open(p, "w", encoding="utf-8").write(s)
q = os.path.join(H, "inline_recall.py"); t = open(q, encoding="utf-8").read()
old3 = '''    ans = gen_raw(prompt3, max_tokens, stop=["\nPerson:"]).strip()'''
new3 = '''    ans = gen_raw(prompt3, max_tokens, stop=["\nPerson:", "<|im_end|>", "\n\n"]).strip()
    ans = ans.split("<|im_end|>")[0].strip()'''
assert old3 in t, "inline stop"; t = t.replace(old3, new3)
open(q, "w", encoding="utf-8").write(t)
import ast; ast.parse(s); ast.parse(t); print("patched")
