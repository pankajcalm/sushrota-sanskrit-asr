#!/usr/bin/env python3
"""Patch 2 for app_sushrota.py: add canon() (dedup + word-final म्==anusvāra) and route
norm() and the transcription prefill builder through it."""
import ast, sys
F = "/home/ece/BigDisk/Prathosh/ASR/scripts/app_sushrota.py"
src = open(F).read()

R1_OLD = """    return ''.join(out)
def norm(s):
    s = unicodedata.normalize('NFC', s); s = dedup_marks(s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\\s+', ' ', s).strip()"""
R1_NEW = """    return ''.join(out)
NASAL = re.compile(r'म्(?=\\s|[।॥]|$)')            # word-final म् (coda /m/) == anusvāra
def canon(s):
    return NASAL.sub('ं', dedup_marks(s))
def norm(s):
    s = unicodedata.normalize('NFC', s); s = canon(s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\\s+', ' ', s).strip()"""

R2_OLD = "            text = dedup_marks(''.join(surf(k) for _, k, _ in cur).replace('▁', ' ')).strip()"
R2_NEW = "            text = canon(''.join(surf(k) for _, k, _ in cur).replace('▁', ' ')).strip()"

if "def canon(" in src:
    print("[skip] canon already present"); sys.exit(0)
for name, old, new in [("norm/canon", R1_OLD, R1_NEW), ("flush text", R2_OLD, R2_NEW)]:
    if old not in src: print(f"[FAIL] anchor not found: {name}"); sys.exit(1)
    src = src.replace(old, new, 1); print(f"[ok] patched {name}")
ast.parse(src); open(F, "w").write(src)
print("[done] written + syntax OK")
