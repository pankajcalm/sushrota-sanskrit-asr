#!/usr/bin/env python3
"""Phase 1.5: honest calibration of the GOP scorer before UI.

Phase 1 used RANDOM token substitution (acoustically easy) -> AUC 0.99, an upper bound.
Here we make it honest:
  1. Build a data-driven ACOUSTIC CONFUSION MAP: over correct alignments, for each target
     token record the real token the model most often ranks 2nd in the target's own frames.
     That 2nd-best token is the hardest realistic mispronunciation of it.
  2. HARD substitution: swap tokens to their top confuser (not random). Lower-bound AUC.
  3. AKSHARA aggregation: users see aksharas, not BPE tokens. Aggregate token NORM-GOP to
     aksharas (mean and min over sub-tokens) and re-measure AUC + false-positive rate.
     A real akshara rarely fails on every sub-token, so aggregation should cut the FPR.
Outputs the honest operating point (aggregation + threshold) and the % -correctness formula.
"""
import os, sys, json, re, unicodedata
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from collections import Counter, defaultdict
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as na

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
OFF, V, BL = 4096, 256, 5632
NEG = -1e30
N_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SUB_RATE = 0.30
SEED = 1234

KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def lse(x, ax):
    mx = x.max(ax, keepdims=True); return mx + np.log(np.exp(x - mx).sum(ax, keepdims=True))

def is_base(ch):
    o = ord(ch)
    return (0x0905 <= o <= 0x0939) or (0x0958 <= o <= 0x0961) or (0x0972 <= o <= 0x097F)
def split_aksharas(chars):
    """chars: list of (char, token_index or -1 for space). Returns list of aksharas, each a
    list of token indices that compose it (dedup, spaces excluded)."""
    aks = []; cur = []; prev_vir = False
    for ch, tj in chars:
        if ch == ' ':
            if cur: aks.append(cur); cur = []
            prev_vir = False; continue
        if is_base(ch) and cur and not prev_vir:
            aks.append(cur); cur = [tj]
        else:
            cur.append(tj)
        prev_vir = (ord(ch) == 0x094D)
    if cur: aks.append(cur)
    # dedup token indices per akshara, drop -1
    return [sorted({t for t in a if t >= 0}) for a in aks]

print("[boot] loading v5", flush=True)
M = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cuda:0").eval()
SUB = M.tokenizer.tokenizers_dict["sa"]
LAB = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
_surf2col = {LAB[d]: d + 1 for d in range(min(V, len(LAB)))}
_CODA = {'ं', 'ँ', 'म्', 'न्', 'ण्', 'ञ्', 'ङ्'}; _CODACOLS = [c for s, c in _surf2col.items() if s in _CODA]
def _is_nasal(s): return s in _CODA
EQUIV = [np.array([c], dtype=int) for c in range(V + 1)]
for d in range(min(V, len(LAB))):
    c = d + 1; s = LAB[d]; eq = {c}
    sib = s[1:] if s.startswith('▁') else '▁' + s
    if sib in _surf2col: eq.add(_surf2col[sib])
    if _is_nasal(s): eq.update(_CODACOLS)
    EQUIV[c] = np.array(sorted(eq), dtype=int)

def posteriors(wav):
    sig = torch.tensor(wav).unsqueeze(0).cuda(); sl = torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
        lp = M.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    cols = [BL] + list(range(OFF, OFF + V)); P = lp[:, cols]
    return P - lse(P, 1)

def align_path(P, ids):
    """Viterbi force-align. Returns (ext, path, L) or None."""
    T = P.shape[0]
    cols = [d + 1 for d in ids if 0 <= d < V]; L = len(cols)
    if L == 0 or T < L: return None
    ext = [0]
    for c in cols: ext += [c, 0]
    ext = np.array(ext); S = len(ext)
    frame_lp = P[:, ext].astype(float, copy=True)
    for k in range(S):
        if ext[k] != 0: frame_lp[:, k] = P[:, EQUIV[ext[k]]].max(1)
    allow = np.zeros(S, bool); allow[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), np.int32)
    dp[0, 0] = frame_lp[0, 0]
    if S > 1: dp[0, 1] = frame_lp[0, 1]
    for t in range(1, T):
        stay = dp[t - 1]
        prev = np.full(S, NEG); prev[1:] = dp[t - 1, :-1]
        skip = np.full(S, NEG); skip[2:] = dp[t - 1, :-2]; skip[~allow] = NEG
        cand = np.stack([stay, prev, skip]); arg = cand.argmax(0)
        dp[t] = cand.max(0) + frame_lp[t]; bp[t] = np.arange(S) - arg
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    if dp[T - 1, s] <= NEG / 2: return None
    path = np.empty(T, np.int32)
    for t in range(T - 1, -1, -1): path[t] = s; s = bp[t, s]
    return ext, path, L, frame_lp

def token_norm_gop(P, ids):
    """per-token (norm_gop, nframes). norm = mean(target_eff - top_logpost) over frames, where
    target_eff maxes over the token's equivalence set (word-boundary + nasal-coda invariance)."""
    r = align_path(P, ids)
    if r is None: return None
    ext, path, L, frame_lp = r; topcol = P[:, 1:].max(1)   # top over REAL tokens (exclude blank)
    out = []
    for j in range(L):
        fr = np.where(path == 2 * j + 1)[0]
        if len(fr) == 0: out.append((NEG, 0))
        else: out.append((float((frame_lp[fr, 2 * j + 1] - topcol[fr]).mean()), len(fr)))
    return out

# ---- data ----
last = {}
for l in open(f"{ROOT}/data/epgp/annot/annot_refs.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    last[r["id"]] = r
pairs = []
for r in last.values():
    if not r.get("text", "").strip() or r.get("unclear"): continue
    p = f"{ROOT}/data/epgp/annot_clips/{r['id']}.wav"; t = norm(r["text"])
    if os.path.exists(p) and t: pairs.append((r["id"], p, t))
pairs.sort(key=lambda x: x[0])
rng = np.random.RandomState(SEED); rng.shuffle(pairs); pairs = pairs[:N_CLIPS]

cache = []
for cid, p, t in pairs:
    wav, _ = sf.read(p, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if len(wav) < 1600: continue
    toks = SUB.text_to_tokens(t); ids = SUB.text_to_ids(t)
    if len(toks) != len(ids): continue
    # char -> token-index stream for akshara mapping
    chars = []
    for j, s in enumerate(toks):
        for ch in s:
            chars.append((' ', -1) if ch == '▁' else (ch, j))
    cache.append((cid, posteriors(wav), ids, split_aksharas(chars)))
print(f"[data] {len(cache)} clips cached", flush=True)

# ---- 1. acoustic confusion map ----
confus = defaultdict(Counter)
for cid, P, ids, aks in cache:
    r = align_path(P, ids)
    if r is None: continue
    ext, path, L, _ = r
    for j in range(L):
        tgt_col = ext[2 * j + 1]
        for t in np.where(path == 2 * j + 1)[0]:
            row = P[t].copy(); row[tgt_col] = NEG; row[0] = NEG    # exclude target + blank
            comp = int(row.argmax())                               # competing column (=id+1)
            if comp >= 1: confus[tgt_col - 1][comp - 1] += 1       # store as sa ids
def top_confuser(tid):
    c = confus.get(tid)
    if not c: return None
    return c.most_common(1)[0][0]
covered = sum(1 for tid in range(V) if confus.get(tid))
print(f"[confusion] built for {covered}/{V} tokens", flush=True)

# ---- 2/3. HARD substitution + akshara aggregation ----
tok_clean, tok_sub = [], []
ak_clean_mean, ak_sub_mean, ak_clean_min, ak_sub_min = [], [], [], []
n_fallback = 0
for cid, P, ids, aks in cache:
    if len(ids) < 4: continue
    ids2 = list(ids); corrupt = set()
    k = max(1, int(round(SUB_RATE * len(ids))))
    for pos in rng.choice(len(ids), size=k, replace=False):
        eqids = set(int(x) - 1 for x in EQUIV[ids2[pos] + 1])   # exclude equivalents (siblings/codas):
        alt = top_confuser(ids2[pos])                           # a same-sound swap is NOT an error
        if alt is None or alt in eqids:
            alt = int(rng.randint(1, V)); n_fallback += 1
            while alt in eqids: alt = int(rng.randint(1, V))
        ids2[pos] = alt; corrupt.add(pos)
    g = token_norm_gop(P, ids2)
    if g is None: continue
    for j, (gop, nf) in enumerate(g):
        (tok_sub if j in corrupt else tok_clean).append(gop)
    # akshara aggregation (token index -> its gop/frames); aksharas indexed against ORIGINAL
    # token layout (substitution keeps positions), corrupted akshara = contains a corrupt token
    gd = {j: g[j] for j in range(len(g))}
    for a in aks:
        vals = [gd[j][0] for j in a if j in gd and gd[j][1] > 0]
        if not vals: continue
        is_corr = any(j in corrupt for j in a)
        m = float(np.mean(vals)); mn = float(np.min(vals))
        (ak_sub_mean if is_corr else ak_clean_mean).append(m)
        (ak_sub_min if is_corr else ak_clean_min).append(mn)

def clean(a): a = np.asarray(a, float); return a[a > NEG / 2]
def auc(pos_low, neg_high):
    pos_low, neg_high = clean(pos_low), clean(neg_high)
    y = np.r_[np.ones(len(pos_low)), np.zeros(len(neg_high))]
    s = np.r_[-pos_low, -neg_high]
    order = np.argsort(s, kind="mergesort"); r = np.empty_like(order, float); r[order] = np.arange(len(s))
    npos = y.sum(); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg)
def report(name, clean_ok, corr):
    ok, cr = clean(clean_ok), clean(corr)
    ps = [5, 10, 25, 50]
    print(f"\n--- {name} ---")
    print("  correct   : " + "  ".join(f"p{p}={np.percentile(ok,p):.2f}" for p in ps) + f"  n={len(ok)}")
    print("  mispron   : " + "  ".join(f"p{p}={np.percentile(cr,p):.2f}" for p in [50,75,90]) + f"  n={len(cr)}")
    print(f"  AUC = {auc(cr, ok):.3f}")
    for fpr in (0.05, 0.10):
        thr = np.percentile(ok, 100 * fpr)
        print(f"  thr@FPR{int(fpr*100)}%={thr:.2f} -> flags {100*(ok<thr).mean():.1f}% correct, catches {100*(cr<thr).mean():.1f}% mispron")

print(f"\n[hard-sub] fallback-to-random substitutions: {n_fallback}")
report("TOKEN level (hard confusable)", tok_clean, tok_sub)
report("AKSHARA level  MEAN aggregation", ak_clean_mean, ak_sub_mean)
report("AKSHARA level  MIN aggregation",  ak_clean_min,  ak_sub_min)
print("\nGOP PHASE15 DONE", flush=True)
