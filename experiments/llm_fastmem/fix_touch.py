import os, re, sys
H = sys.argv[1]
p = os.path.join(H, "fmem_tool.py"); s = open(p, encoding="utf-8").read()
helper = '''

_mem_mtime = [0.0]


def touch_mem():
    """the engine reloads memory.bin when its whole-second mtime changes: make every update strictly later, even
    two updates within one second (e.g. an update wipe followed by the new write)"""
    t = max(time.time(), _mem_mtime[0] + 1.0)
    _mem_mtime[0] = t
    os.utime(MEM, (t, t))
'''
if "def touch_mem" not in s:
    i = s.index("\ndef ")
    s = s[:i] + helper + s[i:]
s = s.replace("os.utime(MEM)", "touch_mem()")
open(p, "w", encoding="utf-8").write(s)
for name in ("fmem_anchor.py", "sleep.py", "exp_b.py"):
    q = os.path.join(H, name); t = open(q, encoding="utf-8").read()
    t = t.replace("os.utime(f.MEM)", "f.touch_mem()")
    open(q, "w", encoding="utf-8").write(t)
print("done")
