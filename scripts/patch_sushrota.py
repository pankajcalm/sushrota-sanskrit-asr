#!/usr/bin/env python3
"""Patch app_sushrota.py: add dedup_marks() and apply it to norm() + the transcription output
builder, so the CTC doubled-combining-mark artifact (सुु -> सु) is fixed in annotation prefills."""
import ast, sys
F = "/home/ece/BigDisk/Prathosh/ASR/scripts/app_sushrota.py"
src = open(F).read()

R1_OLD = """def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\\s+', ' ', s).strip()"""
R1_NEW = """def dedup_marks(s):
    # collapse consecutive identical combining marks — CTC greedy [mātrā,blank,mātrā] artifact
    out = []
    for ch in s:
        if out and ch == out[-1] and unicodedata.category(ch) in ('Mn', 'Mc'): continue
        out.append(ch)
    return ''.join(out)
def norm(s):
    s = unicodedata.normalize('NFC', s); s = dedup_marks(s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\\s+', ' ', s).strip()"""

R2_OLD = "            text = ''.join(surf(k) for _, k, _ in cur).replace('▁', ' ').strip()"
R2_NEW = "            text = dedup_marks(''.join(surf(k) for _, k, _ in cur).replace('▁', ' ')).strip()"

for name, old, new in [("norm/dedup", R1_OLD, R1_NEW), ("flush text", R2_OLD, R2_NEW)]:
    if new.split("\n")[0] in src and name == "norm/dedup" and "def dedup_marks" in src:
        print("[skip] dedup_marks already present"); continue
    if old not in src:
        print(f"[FAIL] anchor not found: {name}"); sys.exit(1)
    src = src.replace(old, new, 1)
    print(f"[ok] patched {name}")

ast.parse(src)                                    # validate before writing
open(F, "w").write(src)
print("[done] written + syntax OK")
