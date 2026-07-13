#!/usr/bin/env python3
"""V5 + shastra-specific lexicon rescoring: WER/CER comparison on a shastra eval.
Baseline (greedy CTC) vs BLIND lexicon projection vs ACOUSTICALLY-GATED (CTC span rescore).
Uses the exact Su-shrotaa deployable mechanism (sa-slice, ctc_score span forward).

usage: eval_sastra_rescore.py <manifest.jsonl> <sastra>  [defaults: purana eval]
"""
import os, sys, json, re, unicodedata, pickle
from collections import Counter, defaultdict
import numpy as np, soundfile as sf, torch
from rapidfuzz import process, distance as rfdist
import nemo.collections.asr as nemo_asr

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
OFF, V, BLANKCOL = 4096, 256, 0
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
MANIFEST = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl"
SASTRA   = sys.argv[2] if len(sys.argv) > 2 else "purana"

KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
LEVd = rfdist.Levenshtein.distance
def ed(a, b): return LEVd(a, b)          # works on strings and on lists of words
def lse(a, ax): m = a.max(ax, keepdims=True); return m + np.log(np.exp(a - m).sum(ax, keepdims=True))

print(f"[boot] model on {DEV} ...", flush=True)
M = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location=DEV)
M.eval()
LABELS = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
def surf(t): return LABELS[t]
_tk = {}
def toks(w):
    if w not in _tk: _tk[w] = M.tokenizer.text_to_ids(w, "sa")
    return _tk[w]

print(f"[boot] loading {SASTRA} lexicon ...", flush=True)
D = pickle.load(open(f"{ROOT}/data/gretil/lex/gretil_by_sastra.pkl", "rb"))
raw = D[SASTRA]
LEXC = Counter()
for w, c in (raw.items() if hasattr(raw, "items") else ((x, 1) for x in raw)):
    wn = norm(w)
    if wn and len(wn) >= 3: LEXC[wn] += int(c) if c else 1
LEX = set(LEXC)
BYLEN = defaultdict(list)
for w in LEX: BYLEN[len(w)].append(w)
print(f"[boot] {SASTRA} lexicon {len(LEX)} forms. eval={MANIFEST}", flush=True)

def cands(h):
    pool = []
    for L in range(len(h) - 2, len(h) + 3): pool.extend(BYLEN.get(L, ()))
    if not pool: return []
    return [(w, d) for w, d, _ in process.extract(h, pool, scorer=LEVd, score_cutoff=2, limit=40) if w != h]

def sa_logprobs(wav):
    sig = torch.tensor(wav).unsqueeze(0).to(DEV); sl = torch.tensor([len(wav)]).to(DEV)
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
        lp = M.ctc_decoder(encoder_output=enc)[0].detach().cpu().numpy()
    cols = [5632] + list(range(OFF, OFF + V))
    sub = lp[:, cols]; return sub - lse(sub, 1)

def ctc_score(seq, P):
    T, L = P.shape[0], len(seq)
    if L == 0 or T == 0: return -1e30
    ext = [BLANKCOL]
    for s in seq: ext += [s + 1, BLANKCOL]
    S = len(ext); NEG = -1e30
    a = np.full(S, NEG); a[0] = P[0, ext[0]]
    if S > 1: a[1] = P[0, ext[1]]
    for t in range(1, T):
        na = np.full(S, NEG)
        for s in range(S):
            v = a[s]
            if s > 0: v = np.logaddexp(v, a[s - 1])
            if s > 1 and ext[s] != BLANKCOL and ext[s] != ext[s - 2]: v = np.logaddexp(v, a[s - 2])
            na[s] = v + P[t, ext[s]]
        a = na
    return float(np.logaddexp(a[S - 1], a[S - 2])) if S > 1 else float(a[S - 1])

def words_spans(P):
    ids = P.argmax(1); emitted = []; prev = -1
    for t, i in enumerate(ids):
        i = int(i)
        if i != prev and i != BLANKCOL: emitted.append((t, i - 1, float(P[t, i])))
        prev = i
    words = []; cur = []
    def flush(end):
        if cur:
            text = ''.join(surf(k) for _, k, _ in cur).replace('▁', ' ').strip()
            words.append({"text": text, "start": cur[0][0], "end": end})
    for (t, k, lp) in emitted:
        if surf(k).startswith('▁') and cur: flush(t); cur = []
        cur.append((t, k, lp))
    flush(len(ids))
    return words

def correct(words, P, gated):
    out = []; nrep = 0
    for w in words:
        h = w["text"]
        if not h: continue
        if h in LEX or len(h) < 3:
            out.append(h); continue
        cs = cands(h)
        if not cs:
            out.append(h); continue
        span = P[w["start"]:max(w["start"] + 1, w["end"])]
        if gated:
            raw_sc = ctc_score(toks(h), span)
            best = max(cs, key=lambda c: ctc_score(toks(c[0]), span))
            if ctc_score(toks(best[0]), span) > raw_sc:
                out.append(best[0]); nrep += 1
            else:
                out.append(h)
        else:
            mind = min(d for _, d in cs)
            best = max([w2 for w2, d in cs if d == mind], key=lambda x: LEXC[x])
            out.append(best); nrep += 1
    return " ".join(out), nrep

rows = [json.loads(l) for l in open(MANIFEST)]
ce = {'base': 0, 'blind': 0, 'gated': 0}; we = {'base': 0, 'blind': 0, 'gated': 0}
cn = 0; wn = 0; reps = {'blind': 0, 'gated': 0}
for idx, r in enumerate(rows):
    wav, sr = sf.read(r['audio_filepath'], dtype='float32')
    if wav.ndim > 1: wav = wav.mean(1)
    P = sa_logprobs(wav)
    words = words_spans(P)
    base = " ".join(w["text"] for w in words)
    blind, nb = correct(words, P, False)
    gated, ng = correct(words, P, True)
    reps['blind'] += nb; reps['gated'] += ng
    rn = norm(r['text']); rc = rn.replace(' ', ''); rw = rn.split()
    cn += len(rc); wn += len(rw)
    for tag, hyp in (('base', base), ('blind', blind), ('gated', gated)):
        hn = norm(hyp)
        ce[tag] += ed(hn.replace(' ', ''), rc)
        we[tag] += ed(hn.split(), rw)
    if (idx + 1) % 100 == 0: print(f".. {idx + 1}/{len(rows)}", flush=True)

print(f"\n=== V5 + {SASTRA} lexicon rescoring | {len(rows)} clips ===")
print(f"{'mode':6s}  {'CER':>7s}   {'WER':>7s}")
for tag in ('base', 'blind', 'gated'):
    print(f"{tag:6s}  {100*ce[tag]/cn:6.2f}%   {100*we[tag]/wn:6.2f}%")
print(f"replacements: blind={reps['blind']}  gated={reps['gated']}  (over {wn} ref words)")
