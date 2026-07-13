#!/usr/bin/env python3
"""Diagnose र-family deletion and sweep a CTC blank penalty. Cache sa-logprobs once per clip,
then for each penalty subtract it from the blank column before greedy argmax. Report CER +
र-family recall (alignment-based) + र over-insertion ratio."""
import os, sys, json, re, unicodedata
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as na
ROOT = "/home/ece/BigDisk/Prathosh/ASR"; OFF, V, BL = 4096, 256, 5632
MODEL = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo"
MANI = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl"
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
RFAM = set("रृॄऋॠऱ")
KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def ed(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    p = list(range(m + 1))
    for i in range(1, n + 1):
        c = [i] + [0] * m; ai = a[i - 1]
        for j in range(1, m + 1): c[j] = min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ai != b[j - 1]))
        p = c
    return p[m]
def r_recall(hyp, ref):
    """char-align ref vs hyp; count ref र-family chars that are matched (recovered)."""
    R = list(ref.replace(' ', '')); H = list(hyp.replace(' ', '')); n, m = len(R), len(H)
    tot = sum(1 for ch in R if ch in RFAM)
    if tot == 0: return 0, 0
    dp = [[0]*(m+1) for _ in range(n+1)]; bt = [['']*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0]=i; bt[i][0]='D'
    for j in range(1, m+1): dp[0][j]=j; bt[0][j]='I'
    for i in range(1, n+1):
        for j in range(1, m+1):
            if R[i-1]==H[j-1]: dp[i][j]=dp[i-1][j-1]; bt[i][j]='M'
            else:
                s,d,ins=dp[i-1][j-1]+1,dp[i-1][j]+1,dp[i][j-1]+1; b=min(s,d,ins); dp[i][j]=b
                bt[i][j]='S' if b==s else ('D' if b==d else 'I')
    matched=0; i,j=n,m
    while i>0 or j>0:
        op=bt[i][j]
        if op=='M':
            if R[i-1] in RFAM: matched+=1
            i-=1; j-=1
        elif op=='S': i-=1; j-=1
        elif op=='D': i-=1
        else: j-=1
    return matched, tot
def lse(x, ax): mx = x.max(ax, keepdims=True); return mx + np.log(np.exp(x - mx).sum(ax, keepdims=True))
LAB = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
M = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL, map_location="cuda:0").eval()
def logp(wav):
    sig = torch.tensor(wav).unsqueeze(0).cuda(); sl = torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
        lp = M.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    cols = [BL] + list(range(OFF, OFF + V)); P = lp[:, cols]; return (P - lse(P, 1)).astype(np.float32)
def decode(P, beta):
    Pb = P.copy(); Pb[:, 0] -= beta; ids = Pb.argmax(1); out = []; prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != 0: out.append(LAB[i - 1])
        prev = i
    return ''.join(out).replace('▁', ' ').strip()
rows = [json.loads(l) for l in open(MANI)]
cache = []
for k, r in enumerate(rows):
    wav, _ = sf.read(r["audio_filepath"], dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    cache.append((logp(wav), norm(r["text"])))
    if (k + 1) % 200 == 0: print(f".. logprobs {k+1}/{len(rows)}", flush=True)
print(f"\n=== {os.path.basename(MODEL)} on {os.path.basename(MANI)} ({len(rows)} clips) ===")
print(f"{'blankpen':>8} {'CER':>7} {'r-recall':>9} {'r-hyp/ref':>10}")
for beta in BETAS:
    ce = cn = rm = rt = rh = 0
    for P, rn in cache:
        hn = norm(decode(P, beta))
        ce += ed(hn.replace(' ', ''), rn.replace(' ', '')); cn += len(rn.replace(' ', ''))
        m_, t_ = r_recall(hn, rn); rm += m_; rt += t_
        rh += sum(1 for ch in hn if ch in RFAM)
    print(f"{beta:>8.1f} {100*ce/cn:>6.2f}% {100*rm/rt:>8.1f}% {rh/max(1,rt):>10.2f}", flush=True)
