#!/usr/bin/env python3
"""Definitive pseudo-label round: v9a (better teacher) transcribes the unused e-PG pool,
keep only utterances where EVERY word conf>0.9 AND v5∩v9a agree (word-editdist<=0.10).
Emit v9a text as the label. Excludes human-labeled clips."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import json, re, unicodedata
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as na
ROOT = "/home/ece/BigDisk/Prathosh/ASR"; OFF, V, BL = 4096, 256, 5632
CONF, DIS_MAX = 0.90, 0.10
KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def wed(a, b):
    a, b = a.split(), b.split(); n, m = len(a), len(b)
    if n == 0: return m / max(1, m)
    if m == 0: return 1.0
    p = list(range(m + 1))
    for i in range(1, n + 1):
        c = [i] + [0] * m
        for j in range(1, m + 1): c[j] = min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (a[i-1] != b[j-1]))
        p = c
    return p[m] / max(n, m)
def lse(x, ax): mx = x.max(ax, keepdims=True); return mx + np.log(np.exp(x - mx).sum(ax, keepdims=True))
LABELS = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
def load(p): m = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(p, map_location="cuda:0"); m.eval(); return m
print("[boot] v9a + v5", flush=True)
V9A = load(f"{ROOT}/exp/ft_ctc_v9a/ft_ctc_ep12.nemo"); V5 = load(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo")
def sa_lp(m, wav):
    sig = torch.tensor(wav).unsqueeze(0).cuda(); sl = torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc, _ = m.forward(input_signal=sig, input_signal_length=sl)
        lp = m.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    cols = [BL] + list(range(OFF, OFF + V)); P = lp[:, cols]; return P - lse(P, 1)
def analyze(m, wav, want_conf):
    P = sa_lp(m, wav); ids = P.argmax(1); em = []; prev = -1
    for t, i in enumerate(ids):
        i = int(i)
        if i != prev and i != 0: em.append((i - 1, float(P[t, i])))
        prev = i
    words = []; cur = []
    for k, lp in em:
        s = LABELS[k]
        if s.startswith('▁') and cur: words.append(cur); cur = []
        cur.append((s, lp))
    if cur: words.append(cur)
    texts = []; confs = []
    for w in words:
        txt = ''.join(s for s, _ in w).replace('▁', ' ').strip()
        if not txt: continue
        texts.append(txt); confs.append(float(np.exp(np.mean([lp for _, lp in w]))))
    return ' '.join(texts), (min(confs) if (want_conf and confs) else 1.0)
# exclude human-labeled ids
labeled = set()
for l in open(f"{ROOT}/data/epgp/annot/annot_refs.jsonl"):
    try: labeled.add(json.loads(l)["id"])
    except: pass
for l in open(f"{ROOT}/data/epgp/v9_heldout.jsonl"):
    labeled.add(os.path.basename(json.loads(l)["audio_filepath"])[:-4])
pool = [json.loads(l) for l in open(f"{ROOT}/data/epgp/manifest_train.jsonl")]
pool = [r for r in pool if 2.0 <= r.get("dur", 0) <= 18.0
        and os.path.basename(r["audio_filepath"])[:-4] not in labeled]
print(f"[pool] {len(pool)} candidate clips (excluded {len(labeled)} labeled)", flush=True)
kept = []; nlow = ndis = 0
with open(f"{ROOT}/data/epgp/pseudo.jsonl", "w") as f:
    for k, r in enumerate(pool):
        try: wav, _ = sf.read(r["audio_filepath"], dtype="float32")
        except Exception: continue
        if wav.ndim > 1: wav = wav.mean(1)
        t9, c9 = analyze(V9A, wav, True)
        if not t9 or c9 < CONF: nlow += 1; continue
        t5, _ = analyze(V5, wav, False)
        if wed(norm(t9), norm(t5)) > DIS_MAX: ndis += 1; continue
        f.write(json.dumps({"audio_filepath": r["audio_filepath"], "text": t9,
                            "duration": r["dur"], "lang": "sa"}, ensure_ascii=False) + "\n")
        kept.append(r)
        if (k + 1) % 2000 == 0: print(f".. {k+1}/{len(pool)} kept {len(kept)}", flush=True)
hrs = sum(r["dur"] for r in kept) / 3600
print(f"\nPSEUDO DONE: kept {len(kept)}/{len(pool)} ({hrs:.1f}h) | dropped lowconf {nlow} disagree {ndis}", flush=True)
