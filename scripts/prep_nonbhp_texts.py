#!/usr/bin/env python3
"""Parse pasted/fetched non-BhP texts -> per-chapter normalized verse JSONs."""
import os, re, json, unicodedata
SCR="/private/tmp/claude-501/-Users-prathosh-ASR/768c602d-fa6f-4371-932b-d230dee0334b/scratchpad"
OUT="/tmp/nonbhp"; os.makedirs(OUT, exist_ok=True)
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ]')
def norm(s):
    s=unicodedata.normalize('NFC', s); s=DROP.sub(' ', s); s=KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
VMARK=re.compile(r'॥\s*[०-९\d]+\s*॥')      # verse-end marker w/ number (Deva or Arabic digits)

# --- Vishnu Sahasranama (single text) ---
raw=open(f"{SCR}/vsn_raw.txt").read()
verses=[norm(v) for v in VMARK.split(raw) if norm(v)]
json.dump(verses, open(f"{OUT}/vishnu_sahasranama.json","w"), ensure_ascii=False)
print(f"vishnu_sahasranama: {len(verses)} verses")

# --- Dvadasha Stotra (12 chapters) ---
dv=open(f"{SCR}/dvadasha_raw.txt").read()
chaps=re.split(r'===\s*CHAPTER\s*(\d+)\s*===', dv)
# chaps = ['', '1', <text1>, '2', <text2>, ...]
for i in range(1, len(chaps), 2):
    ch=int(chaps[i]); body=chaps[i+1]
    vs=[norm(v) for v in VMARK.split(body) if norm(v)]
    json.dump(vs, open(f"{OUT}/dvadasha_stotra_{ch}.json","w"), ensure_ascii=False)
    print(f"dvadasha ch{ch}: {len(vs)} verses")
