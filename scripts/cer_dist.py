#!/usr/bin/env python3
"""CER distribution + pattern analysis: per-clip CER percentiles/histogram, per-speaker,
per-duration, and character-level error patterns (top deletions + substitution pairs)."""
import os, sys, json, re, unicodedata
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from collections import Counter, defaultdict
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as na
ROOT = "/home/ece/BigDisk/Prathosh/ASR"; OFF, V, BL = 4096, 256, 5632
MODEL = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo"
KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def align_ops(ref, hyp, dele, sub):
    """char-align; tally deletions (ref char dropped) and substitutions (ref->hyp)."""
    R = list(ref); H = list(hyp); n, m = len(R), len(H)
    dp = [[0]*(m+1) for _ in range(n+1)]; bt = [['']*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0]=i; bt[i][0]='D'
    for j in range(1, m+1): dp[0][j]=j; bt[0][j]='I'
    for i in range(1, n+1):
        for j in range(1, m+1):
            if R[i-1]==H[j-1]: dp[i][j]=dp[i-1][j-1]; bt[i][j]='M'
            else:
                s_,d_,ins_=dp[i-1][j-1]+1,dp[i-1][j]+1,dp[i][j-1]+1; b=min(s_,d_,ins_); dp[i][j]=b
                bt[i][j]='S' if b==s_ else ('D' if b==d_ else 'I')
    i,j=n,m
    while i>0 or j>0:
        o=bt[i][j]
        if o=='M': i-=1;j-=1
        elif o=='S': sub[(R[i-1],H[j-1])]+=1; i-=1;j-=1
        elif o=='D': dele[R[i-1]]+=1; i-=1
        else: j-=1
    return dp[n][m]
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
def spk_of(r):
    return r.get("speaker") or os.path.basename(r["audio_filepath"]).rsplit("_", 1)[0]
SETS = [("chant", f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl"),
        ("heldout", f"{ROOT}/data/epgp/v9_heldout.jsonl")]
for name, mani in SETS:
    if not os.path.exists(mani): continue
    cers = []; spk = defaultdict(lambda: [0, 0]); durb = defaultdict(lambda: [0, 0])
    dele = Counter(); sub = Counter(); TE = TC = 0
    for l in open(mani):
        r = json.loads(l); wav, _ = sf.read(r["audio_filepath"], dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        rn = norm(r["text"]).replace(' ', ''); hn = norm(greedy(wav)).replace(' ', '')
        if not rn: continue
        e = align_ops(rn, hn, dele, sub); c = 100 * e / len(rn)
        cers.append(c); TE += e; TC += len(rn)
        s = spk_of(r); spk[s][0] += e; spk[s][1] += len(rn)
        d = r.get("duration", 0); db = "0-3" if d < 3 else "3-6" if d < 6 else "6-10" if d < 10 else "10+"
        durb[db][0] += e; durb[db][1] += len(rn)
    a = np.array(cers); print(f"\n================ {name} ({len(cers)} clips, micro-CER {100*TE/TC:.2f}%) ================")
    ps = [10, 25, 50, 75, 90, 95, 99]
    print("per-clip CER pctiles: " + "  ".join(f"p{p}={np.percentile(a,p):.1f}%" for p in ps) + f"  mean={a.mean():.1f}%")
    buckets = [(0,2),(2,5),(5,10),(10,20),(20,50),(50,1e9)]
    print("histogram: " + "  ".join(f"[{lo}-{hi if hi<1e9 else '∞'}%):{100*((a>=lo)&(a<hi)).mean():.0f}%" for lo,hi in buckets))
    print("by duration: " + "  ".join(f"{k}:{100*durb[k][0]/max(1,durb[k][1]):.1f}%(n_char={durb[k][1]})" for k in ["0-3","3-6","6-10","10+"] if k in durb))
    worst = sorted(spk.items(), key=lambda kv: -kv[1][0]/max(1,kv[1][1]))
    print("worst speakers: " + "  ".join(f"{s[:10]}:{100*e/max(1,c):.0f}%" for s,(e,c) in worst[:6]))
    print("best  speakers: " + "  ".join(f"{s[:10]}:{100*e/max(1,c):.0f}%" for s,(e,c) in worst[-4:]))
    print("top DELETIONS: " + "  ".join(f"{repr(ch)}:{n}" for ch,n in dele.most_common(10)))
    print("top SUBS(ref→hyp): " + "  ".join(f"{a_}→{b_}:{n}" for (a_,b_),n in sub.most_common(10)), flush=True)
