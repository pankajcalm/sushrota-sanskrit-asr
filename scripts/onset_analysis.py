#!/usr/bin/env python3
"""Quantify the onset artifact: word-error rate by reference-word POSITION. If positions 0-1
are far worse than the rest, the abrupt VAD clip-start is inflating WER. Also reports overall
WER vs WER-excluding-first-2-ref-words (the onset contribution)."""
import os, sys, json, re, unicodedata
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as na
ROOT = "/home/ece/BigDisk/Prathosh/ASR"; OFF, V, BL = 4096, 256, 5632
MODEL = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo"
KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def edw(a, b):
    n, m = len(a), len(b)
    p = list(range(m + 1))
    for i in range(1, n + 1):
        c = [i] + [0]*m
        for j in range(1, m + 1): c[j] = min(p[j]+1, c[j-1]+1, p[j-1]+(a[i-1] != b[j-1]))
        p = c
    return p[m]
def ref_pos_correct(hyp_w, ref_w):
    """word-align; return list over ref positions: 1 if that ref word is matched, else 0."""
    n, m = len(ref_w), len(hyp_w)
    dp = [[0]*(m+1) for _ in range(n+1)]; bt = [['']*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0]=i; bt[i][0]='D'
    for j in range(1, m+1): dp[0][j]=j; bt[0][j]='I'
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref_w[i-1]==hyp_w[j-1]: dp[i][j]=dp[i-1][j-1]; bt[i][j]='M'
            else:
                s,d,ins=dp[i-1][j-1]+1,dp[i-1][j]+1,dp[i][j-1]+1; b=min(s,d,ins); dp[i][j]=b
                bt[i][j]='S' if b==s else ('D' if b==d else 'I')
    ok=[0]*n; i,j=n,m
    while i>0 or j>0:
        o=bt[i][j]
        if o=='M': ok[i-1]=1; i-=1; j-=1
        elif o=='S': i-=1; j-=1
        elif o=='D': i-=1
        else: j-=1
    return ok
def lse(x, ax): mx = x.max(ax, keepdims=True); return mx + np.log(np.exp(x-mx).sum(ax, keepdims=True))
LAB = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
M = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL, map_location="cuda:0").eval()
def greedy(wav):
    sig = torch.tensor(wav).unsqueeze(0).cuda(); sl = torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
        lp = M.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    cols = [BL] + list(range(OFF, OFF+V)); P = lp[:, cols]; P = P - lse(P, 1); ids = P.argmax(1)
    o=[]; prev=-1
    for i in ids:
        i=int(i)
        if i!=prev and i!=0: o.append(LAB[i-1])
        prev=i
    return ''.join(o).replace('▁', ' ').strip()
SETS = [("chant", f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl"),
        ("heldout", f"{ROOT}/data/epgp/v9_heldout.jsonl")]
for name, mani in SETS:
    if not os.path.exists(mani): continue
    err = np.zeros(8); cnt = np.zeros(8)       # error/count by ref position (0..6, 7=rest)
    we = wn = we2 = wn2 = 0
    for l in open(mani):
        r = json.loads(l); wav, _ = sf.read(r["audio_filepath"], dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        hw = norm(greedy(wav)).split(); rw = norm(r["text"]).split()
        if not rw: continue
        ok = ref_pos_correct(hw, rw)
        for i, o in enumerate(ok):
            b = min(i, 7); cnt[b] += 1; err[b] += (1 - o)
        we += edw(hw, rw); wn += len(rw)
        we2 += edw(hw[2:] if len(hw) > 2 else [], rw[2:]); wn2 += max(0, len(rw) - 2)   # drop first 2 ref+hyp words
    print(f"\n=== {name} ({os.path.basename(MODEL)}) ===")
    print("pos:   " + " ".join(f"{i}:{100*err[i]/max(1,cnt[i]):4.0f}%" for i in range(7)) + f"  rest:{100*err[7]/max(1,cnt[7]):4.0f}%")
    print(f"WER all {100*we/wn:.2f}%   WER excl-first-2 {100*we2/max(1,wn2):.2f}%   (onset drag ~{100*we/wn - 100*we2/max(1,wn2):+.2f})", flush=True)
