import ast, os, sys
p = sys.argv[1]
L = open(p, encoding="utf-8").read().split("\n")
i0 = next(i for i, l in enumerate(L) if l.startswith("THINK_SYS = ("))
i1 = next(i for i, l in enumerate(L) if i > i0 and "Always start your reply with the anchor line." in l)
new = [
    'THINK_SYS = ("You are Aria\'s thinking mind. The line you get starts with the speaker\'s name. You have no notes. To "',
    '             "remember where a person\'s thing is, write the anchor [owner|object] -> using the owner\'s name and the bare "',
    '             "object noun, and the place comes to you right after the arrow. If nothing comes after the arrow, say you do "',
    '             "not remember. The words after the arrow are your own memory: trust them and use them.\n\n"',
    '             "Examples of how you start:\n"',
    '             \'Ken says: "ugh where did I leave my charger again??" -> you write: [Ken|charger] ->\n\'',
    '             \'Mia says: "Aria, my wallet?" -> you write: [Mia|wallet] ->\n\'',
    '             \'Tom says: "did I ever tell you where my ring is?" -> you write: [Tom|ring] ->\n\'',
    '             \'Sara says: "any idea where Leo put his scarf? he is freaking out" -> you write: [Leo|scarf] ->\n\'',
    '             "Always start your reply with the anchor line.")',
]
L[i0:i1 + 1] = new
s = "\n".join(L)
ast.parse(s)
open(p, "w", encoding="utf-8").write(s)
print("fixed")
