#!/usr/bin/env python3
"""Sandhi/segmentation-normalized WER: how much of V5's WER is real word errors vs
word-boundary (sandhi) cosmetics. Strip spaces from hyp+ref, align at CHARACTER level,
and count a reference word correct iff every one of its characters is recovered — regardless
of where boundaries fall. Reports standard CER/WER (sanity) + space-insensitive WER.

usage: eval_sandhi_wer.py <manifest.jsonl>
"""
import os, sys, json, re, unicodedata
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as nemo_asr

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
OFF, V, BLANKCOL = 4096, 256, 0
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
MANIFEST = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl"

KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def lse(a, ax): m = a.max(ax, keepdims=True); return m + np.log(np.exp(a - m).sum(ax, keepdims=True))
def ed(a, b):  # plain Levenshtein on sequences (for the sanity CER/WER numbers)
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]

print(f"[boot] model on {DEV} ...", flush=True)
M = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location=DEV)
M.eval()
LABELS = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
def surf(t): return LABELS[t]

def greedy(wav):
    sig = torch.tensor(wav).unsqueeze(0).to(DEV); sl = torch.tensor([len(wav)]).to(DEV)
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
        lp = M.ctc_decoder(encoder_output=enc)[0].detach().cpu().numpy()
    cols = [5632] + list(range(OFF, OFF + V)); P = lp[:, cols] - lse(lp[:, cols], 1)
    ids = P.argmax(1); out = []; prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != BLANKCOL: out.append(surf(i - 1))
        prev = i
    return ''.join(out).replace('▁', ' ').strip()

def spaceless_word_errors(hyp, ref):
    """Align space-stripped hyp vs ref at char level; a ref word is damaged if any of its
    chars is subbed/deleted (strict also counts insertions attributed to the preceding word).
    Returns (n_words, bad_strict, bad_subdel, n_ins)."""
    rw = ref.split()
    if not rw: return (0, 0, 0, 0)
    R, owner = [], []
    for wi, w in enumerate(rw):
        for ch in w: R.append(ch); owner.append(wi)
    H = list("".join(hyp.split()))
    n, m = len(R), len(H)
    # DP with backtrace
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[''] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0] = i; bt[i][0] = 'D'
    for j in range(1, m + 1): dp[0][j] = j; bt[0][j] = 'I'
    for i in range(1, n + 1):
        Ri = R[i - 1]
        for j in range(1, m + 1):
            if Ri == H[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]; bt[i][j] = 'M'
            else:
                sub, dele, ins = dp[i - 1][j - 1] + 1, dp[i - 1][j] + 1, dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                bt[i][j] = 'S' if best == sub else ('D' if best == dele else 'I')
    # backtrace, attribute ops to ref words
    dmg_strict = [False] * len(rw); dmg_sd = [False] * len(rw); n_ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == 'M':
            i -= 1; j -= 1
        elif op == 'S':
            dmg_strict[owner[i - 1]] = True; dmg_sd[owner[i - 1]] = True; i -= 1; j -= 1
        elif op == 'D':
            dmg_strict[owner[i - 1]] = True; dmg_sd[owner[i - 1]] = True; i -= 1
        else:  # 'I' — extra hyp char, attribute to the ref word at/just before this point
            w = owner[i - 1] if i >= 1 else owner[0]
            dmg_strict[w] = True; n_ins += 1; j -= 1
    return (len(rw), sum(dmg_strict), sum(dmg_sd), n_ins)

rows = [json.loads(l) for l in open(MANIFEST)]
ce = we = cn = wn = 0
W = 0; BAD_S = 0; BAD_SD = 0; INS = 0
for idx, r in enumerate(rows):
    wav, sr = sf.read(r['audio_filepath'], dtype='float32')
    if wav.ndim > 1: wav = wav.mean(1)
    hn = norm(greedy(wav)); rn = norm(r['text'])
    ce += ed(hn.replace(' ', ''), rn.replace(' ', '')); cn += len(rn.replace(' ', ''))
    we += ed(hn.split(), rn.split()); wn += len(rn.split())
    nw, bs, bsd, ni = spaceless_word_errors(hn, rn)
    W += nw; BAD_S += bs; BAD_SD += bsd; INS += ni
    if (idx + 1) % 100 == 0: print(f".. {idx + 1}/{len(rows)}", flush=True)

avglen = cn / wn
print(f"\n=== V5 on {os.path.basename(MANIFEST)} | {len(rows)} clips, {wn} ref words, avg word {avglen:.2f} chars ===")
print(f"standard          CER {100*ce/cn:6.2f}%   WER {100*we/wn:6.2f}%")
print(f"space-insensitive WER (subs+dels only)      {100*BAD_SD/W:6.2f}%   <- real word errors, boundaries free")
print(f"space-insensitive WER (strict, +insertions) {100*BAD_S/W:6.2f}%   ({INS} char-insertions attributed)")
print(f"=> segmentation share of standard WER: {100*(we/wn - BAD_SD/W)/(we/wn):.1f}% (lenient) .. {100*(we/wn - BAD_S/W)/(we/wn):.1f}% (strict)")
