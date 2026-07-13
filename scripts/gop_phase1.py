#!/usr/bin/env python3
"""Phase 1: CTC forced-alignment + GOP core, validated OFFLINE on our annotated clips.

Uses v5 exactly as shipped: acoustic encoder + CTC head, sa-slice posteriors
(cols=[BLANK]+range(4096,4352), re-log_softmax). Reference text -> sa SentencePiece
ids -> forced alignment (Viterbi over ref tokens + blank) -> per-token GOP.

Does GOP separate correct chanting from mispronunciation? Three experiments on the
same audio:
  A  CORRECT      : true annotation text vs its own audio      (should score HIGH)
  B  SUBSTITUTION : ~30% of tokens swapped to a random other   (swapped positions LOW,
                    token, audio unchanged                       untouched positions HIGH)
  C  WRONG-AUDIO  : true text vs a DIFFERENT clip's audio       (all tokens LOW; tests the
                                                                 'Viterbi always aligns' caveat)
Reports GOP distributions, mispronunciation-detection AUC, and a threshold at a target
false-positive rate (fraction of genuinely-correct tokens we'd wrongly flag).
"""
import os, sys, json, re, unicodedata
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as na

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
OFF, V, BL = 4096, 256, 5632
NEG = -1e30
N_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SUB_RATE = 0.30
SEED = 1234  # Math.random unavailable in workflows; here plain python, fixed for reproducibility

KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def lse(x, ax):
    mx = x.max(ax, keepdims=True); return mx + np.log(np.exp(x - mx).sum(ax, keepdims=True))

print("[boot] loading v5", flush=True)
M = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cuda:0").eval()
SUB = M.tokenizer.tokenizers_dict["sa"]        # SentencePieceTokenizer; ids index LAB directly
VOCAB = SUB.vocab_size if hasattr(SUB, "vocab_size") else 256

def posteriors(wav):
    """[T,257] log-probs over [blank]+sa-slice, re-log_softmaxed. col0=blank, col d+1 = sa id d."""
    sig = torch.tensor(wav).unsqueeze(0).cuda(); sl = torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
        lp = M.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    cols = [BL] + list(range(OFF, OFF + V)); P = lp[:, cols]
    return P - lse(P, 1)

def force_align(P, ids):
    """Viterbi force-align sa ids to P. Returns per-token (raw_gop, norm_gop, nframes).
      raw  = mean target log-posterior over the token's aligned frames
      norm = mean (target_logpost - max_logpost) over those frames  (0 = target is top choice)
    Returns None if T < required path length (audio too short for the text)."""
    T = P.shape[0]; L = len(ids)
    if L == 0 or T < L: return None
    cols = [d + 1 for d in ids if 0 <= d < V]           # guard: drop the 1 out-of-slice id (256)
    if len(cols) != L: L = len(cols)
    if L == 0 or T < L: return None
    ext = [0]
    for c in cols: ext += [c, 0]
    ext = np.array(ext); S = len(ext)                    # 2L+1
    frame_lp = P[:, ext]                                 # [T,S] log-post of each slot's symbol
    topcol = P.max(1)                                    # [T] best log-post per frame (for norm)
    # skip s-2 -> s allowed only into a non-blank slot whose label differs from label two back
    allow = np.zeros(S, bool)
    allow[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), np.int32)
    dp[0, 0] = frame_lp[0, 0]
    if S > 1: dp[0, 1] = frame_lp[0, 1]
    for t in range(1, T):
        stay = dp[t - 1]
        prev = np.full(S, NEG); prev[1:] = dp[t - 1, :-1]
        skip = np.full(S, NEG); skip[2:] = dp[t - 1, :-2]; skip[~allow] = NEG
        cand = np.stack([stay, prev, skip])              # 3xS: 0=stay,1=s-1,2=s-2
        arg = cand.argmax(0); best = cand.max(0)
        dp[t] = best + frame_lp[t]; bp[t] = np.arange(S) - arg
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    if dp[T - 1, s] <= NEG / 2: return None              # infeasible
    path = np.empty(T, np.int32)
    for t in range(T - 1, -1, -1): path[t] = s; s = bp[t, s]
    out = []
    for j in range(L):
        sj = 2 * j + 1
        fr = np.where(path == sj)[0]
        if len(fr) == 0:
            out.append((NEG, NEG, 0))                     # unreachable in valid CTC path
        else:
            raw = float(frame_lp[fr, sj].mean())
            nrm = float((frame_lp[fr, sj] - topcol[fr]).mean())
            out.append((raw, nrm, len(fr)))
    return out

# ---- load annotated pairs (known-correct text+audio) ----
last = {}
for l in open(f"{ROOT}/data/epgp/annot/annot_refs.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    last[r["id"]] = r
pairs = []
for r in last.values():
    if not r.get("text", "").strip() or r.get("unclear"): continue
    p = f"{ROOT}/data/epgp/annot_clips/{r['id']}.wav"
    t = norm(r["text"])
    if os.path.exists(p) and t: pairs.append((r["id"], p, t))
pairs.sort(key=lambda x: x[0])
rng = np.random.RandomState(SEED); rng.shuffle(pairs)
pairs = pairs[:N_CLIPS]
print(f"[data] {len(pairs)} annotated clips", flush=True)

# precompute posteriors + token ids once
cache = []
for cid, p, t in pairs:
    wav, _ = sf.read(p, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if len(wav) < 1600: continue
    ids = SUB.text_to_ids(t)
    P = posteriors(wav)
    cache.append((cid, P, ids))
print(f"[data] posteriors cached for {len(cache)}", flush=True)

# ---- A: correct ----
A = []
for cid, P, ids in cache:
    r = force_align(P, ids)
    if r: A += r
# ---- B: substitution (corrupt ~30% of positions, same audio) ----
B_clean, B_sub = [], []
for cid, P, ids in cache:
    if len(ids) < 4: continue
    ids2 = list(ids); corrupt = set()
    k = max(1, int(round(SUB_RATE * len(ids))))
    for pos in rng.choice(len(ids), size=k, replace=False):
        alt = int(rng.randint(1, V))                     # any real sa id != current, in-slice
        while alt == ids2[pos]: alt = int(rng.randint(1, V))
        ids2[pos] = alt; corrupt.add(pos)
    r = force_align(P, ids2)
    if not r: continue
    for j, g in enumerate(r):
        (B_sub if j in corrupt else B_clean).append(g)
# ---- C: wrong audio (true text vs next clip's posteriors) ----
C = []
for i, (cid, P, ids) in enumerate(cache):
    Pw = cache[(i + 1) % len(cache)][1]
    r = force_align(Pw, ids)
    if r: C += r

def stats(rows, idx):
    a = np.array([x[idx] for x in rows], float); a = a[a > NEG / 2]
    ps = [5, 10, 25, 50, 75, 90]
    return a, "  ".join(f"p{p}={np.percentile(a,p):.2f}" for p in ps) + f"  mean={a.mean():.2f}  n={len(a)}"

def auc(pos_low, neg_high):
    """AUC for detecting mispronunciation: score = -GOP; label 1 = should-flag."""
    y = np.r_[np.ones(len(pos_low)), np.zeros(len(neg_high))]
    s = np.r_[-np.asarray(pos_low), -np.asarray(neg_high)]
    order = np.argsort(s); r = np.empty_like(order, float); r[order] = np.arange(len(s))
    npos = y.sum(); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg)

for name, idx in [("RAW gop", 0), ("NORM gop (target - top)", 1)]:
    print(f"\n================ {name} ================")
    aA, sA = stats(A, idx);          print(f"  A correct       : {sA}")
    aBc, sBc = stats(B_clean, idx);  print(f"  B clean (untouched): {sBc}")
    aBs, sBs = stats(B_sub, idx);    print(f"  B substituted   : {sBs}")
    aC, sC = stats(C, idx);          print(f"  C wrong-audio   : {sC}")
    au_sub = auc(aBs, aBc); au_wrong = auc(aC, aA)
    print(f"  AUC substitution-vs-clean : {au_sub:.3f}")
    print(f"  AUC wrongaudio-vs-correct : {au_wrong:.3f}")
    # threshold at target FPR on genuinely-correct tokens (A); report recall on substituted
    for fpr in (0.05, 0.10):
        thr = np.percentile(aA, 100 * fpr)               # flag GOP < thr
        rec = float((aBs < thr).mean())
        fp = float((aA < thr).mean())
        print(f"  thr@FPR{int(fpr*100)}%={thr:.2f} -> flags {100*fp:.1f}% correct, catches {100*rec:.1f}% substituted")
print("\nGOP PHASE1 DONE", flush=True)
