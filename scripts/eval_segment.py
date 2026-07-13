#!/usr/bin/env python3
"""Sandhi-aware output re-segmentation (char-preserving). For each ASR token, if it tiles
cleanly into >=2 dictionary words, split it there; else keep it whole (rare compounds safe).
Measures WER before vs after (target: drop toward the SN-WER floor), CER unchanged."""
import os, sys, json, re, unicodedata, pickle, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as na
ROOT = "/home/ece/BigDisk/Prathosh/ASR"; OFF, V, BL = 4096, 256, 5632
MODEL = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo"
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
        c = [i] + [0]*m; ai = a[i-1]
        for j in range(1, m+1): c[j] = min(p[j]+1, c[j-1]+1, p[j-1]+(ai != b[j-1]))
        p = c
    return p[m]
# lexicon -> log-freq
LEXC = pickle.load(open(f"{ROOT}/data/lexicon.pkl", "rb"))
TOT = sum(LEXC.values()); LOGF = {w: math.log(c / TOT) for w, c in LEXC.items()}
LEX = set(LEXC); MAXLEN = 22
def split_token(tok, wpen=6.0, minp=2):
    """Split tok into >=2 dict words iff it fully tiles into known words; else keep whole."""
    n = len(tok)
    if n < 6 or tok in LEX: return [tok]
    NEG = -1e18; dp = [NEG]*(n+1); bp = [-1]*(n+1); dp[0] = 0.0
    for i in range(1, n+1):
        for j in range(max(0, i-MAXLEN), i-minp+1):
            w = tok[j:i]
            if len(w) < minp or w not in LEX or dp[j] == NEG: continue
            sc = dp[j] + LOGF[w] - wpen                # reward freq, penalize each extra word
            if sc > dp[i]: dp[i] = sc; bp[i] = j
    if dp[n] == NEG: return [tok]                       # no all-known tiling -> keep whole (safe)
    words = []; i = n
    while i > 0: words.append(tok[bp[i]:i]); i = bp[i]
    words = words[::-1]
    return words if len(words) > 1 else [tok]
def reseg(text):
    return ' '.join(w for tok in text.split() for w in split_token(tok))
def lse(x, ax): mx = x.max(ax, keepdims=True); return mx + np.log(np.exp(x-mx).sum(ax, keepdims=True))
def spaceless(hyp, ref):  # SN-WER lo (subs+dels)
    rw = ref.split()
    if not rw: return 0, 0
    R = []; own = []
    for wi, w in enumerate(rw):
        for ch in w: R.append(ch); own.append(wi)
    H = list(hyp.replace(' ', '')); n, m = len(R), len(H)
    dp = [[0]*(m+1) for _ in range(n+1)]; bt = [['']*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0]=i; bt[i][0]='D'
    for j in range(1, m+1): dp[0][j]=j; bt[0][j]='I'
    for i in range(1, n+1):
        for j in range(1, m+1):
            if R[i-1]==H[j-1]: dp[i][j]=dp[i-1][j-1]; bt[i][j]='M'
            else:
                s,d,ins = dp[i-1][j-1]+1, dp[i-1][j]+1, dp[i][j-1]+1; b=min(s,d,ins); dp[i][j]=b
                bt[i][j]='S' if b==s else ('D' if b==d else 'I')
    bad=[False]*len(rw); i,j=n,m
    while i>0 or j>0:
        o=bt[i][j]
        if o=='M': i-=1;j-=1
        elif o=='S': bad[own[i-1]]=True;i-=1;j-=1
        elif o=='D': bad[own[i-1]]=True;i-=1
        else: j-=1
    return sum(bad), len(rw)
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
        ("prose", f"{ROOT}/data/utts_fa/manifest_prose_eval_vtn.jsonl"),
        ("heldout", f"{ROOT}/data/epgp/v9_heldout.jsonl")]
print(f"=== {os.path.basename(MODEL)} | sandhi re-segmentation ===")
print(f"{'set':8} {'CER':>6} {'WER base':>9} {'WER seg':>8} {'SN-WER':>7}")
for name, mani in SETS:
    if not os.path.exists(mani): continue
    ce=cn=web=wes=snb=snn=0
    for l in open(mani):
        r = json.loads(l); wav, _ = sf.read(r["audio_filepath"], dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        h = norm(greedy(wav)); rn = norm(r["text"])
        hs = reseg(h)
        ce += ed(h.replace(' ', ''), rn.replace(' ', '')); cn += len(rn.replace(' ', ''))
        web += ed(h.split(), rn.split()); wes += ed(hs.split(), rn.split());
        sb, sd = spaceless(h, rn); snb += sb; snn += sd
    globals().setdefault('wn', 0)
    wn = sum(len(json.loads(l)["text"].split()) for l in open(mani))  # ref words (approx via raw; use norm below)
    wn = 0
    for l in open(mani): wn += len(norm(json.loads(l)["text"]).split())
    print(f"{name:8} {100*ce/cn:>5.2f}% {100*web/wn:>8.2f}% {100*wes/wn:>7.2f}% {100*snb/snn:>6.2f}%", flush=True)
