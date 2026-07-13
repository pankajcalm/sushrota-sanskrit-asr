#!/usr/bin/env python3
"""Benchmark the ByT5-Sanskrit post-corrector on the REAL evals. v5 greedy -> IAST -> ByT5
-> Devanagari; score baseline vs corrected (content-only CER/WER). Prints examples to watch
for rare-term over-correction."""
import os, json, re, unicodedata
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as na
from transformers import AutoTokenizer, T5ForConditionalGeneration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
ROOT = "/home/ece/BigDisk/Prathosh/ASR"; OFF, V, BL = 4096, 256, 5632
BASE = "buddhist-nlp/byt5-sanskrit"; PC = f"{ROOT}/exp/byt5_postcorr/best"; MAXLEN = 384
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
def d2i(s): return transliterate(s, sanscript.DEVANAGARI, sanscript.IAST)
def i2d(s): return transliterate(s, sanscript.IAST, sanscript.DEVANAGARI)
def lse(x, ax): mx = x.max(ax, keepdims=True); return mx + np.log(np.exp(x-mx).sum(ax, keepdims=True))
LAB = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
M = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cuda:0").eval()
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
print("[boot] ByT5 post-corrector", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
pc = T5ForConditionalGeneration.from_pretrained(PC).cuda().eval()
def correct(hyps):
    out = []; B = 16
    for i in range(0, len(hyps), B):
        inp = [d2i(h) for h in hyps[i:i+B]]
        enc = tok(inp, return_tensors='pt', padding=True, truncation=True, max_length=MAXLEN).to('cuda')
        with torch.no_grad(): gen = pc.generate(**enc, max_length=MAXLEN, num_beams=4)
        out += [i2d(tok.decode(g, skip_special_tokens=True)) for g in gen]
    return out
SETS = [("chant", f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl"),
        ("prose", f"{ROOT}/data/utts_fa/manifest_prose_eval_vtn.jsonl"),
        ("heldout", f"{ROOT}/data/epgp/v9_heldout.jsonl")]
for name, mani in SETS:
    if not os.path.exists(mani): continue
    rows = [json.loads(l) for l in open(mani)]
    hyps = []; refs = []
    for r in rows:
        wav, _ = sf.read(r["audio_filepath"], dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        hyps.append(greedy(wav)); refs.append(r["text"])
    corr = correct(hyps)
    ceb=cec=cn=web=wec=wn=0
    for h, c, rf in zip(hyps, corr, refs):
        hn=norm(h); cnn=norm(c); rn=norm(rf)
        ceb += ed(hn.replace(' ',''), rn.replace(' ','')); cec += ed(cnn.replace(' ',''), rn.replace(' ',''))
        web += ed(hn.split(), rn.split()); wec += ed(cnn.split(), rn.split())
        cn += len(rn.replace(' ','')); wn += len(rn.split())
    print(f"\n=== {name} ({len(rows)} clips) ===")
    print(f"  baseline : CER {100*ceb/cn:.2f}%  WER {100*web/wn:.2f}%")
    print(f"  corrected: CER {100*cec/cn:.2f}%  WER {100*wec/wn:.2f}%")
    print("  examples (hyp | corrected | ref):", flush=True)
    for k in range(min(3, len(rows))):
        print(f"    H: {norm(hyps[k])[:60]}")
        print(f"    C: {norm(corr[k])[:60]}")
        print(f"    R: {norm(refs[k])[:60]}", flush=True)
