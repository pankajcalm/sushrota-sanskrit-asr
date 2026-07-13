#!/usr/bin/env python3
"""Diagnose onset clipping: compare v5 on the cut clip vs a version re-sliced from the
original full audio with PAD seconds of lead-in. If the padded version recovers coherent
leading words, the VAD cut the onset and we should re-slice all clips with padding."""
import os, json, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as na
ROOT = "/home/ece/BigDisk/Prathosh/ASR"; OFF, VV, BL = 4096, 256, 5632; PAD = 1.5
M = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cuda:0").eval()
LAB = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
def lse(a, ax): m = a.max(ax, keepdims=True); return m + np.log(np.exp(a - m).sum(ax, keepdims=True))
def greedy(wav):
    sig = torch.tensor(wav).unsqueeze(0).cuda(); sl = torch.tensor([len(wav)]).cuda()
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
        lp = M.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    cols = [BL] + list(range(OFF, OFF + VV)); P = lp[:, cols]; P = P - lse(P, 1); ids = P.argmax(1)
    o = []; prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != 0: o.append(LAB[i - 1])
        prev = i
    return ''.join(o).replace('▁', ' ').strip()
# join hardneg ids -> start/dur/video from manifest_train
tr = {os.path.basename(json.loads(l)["audio_filepath"])[:-4]: json.loads(l) for l in open(f"{ROOT}/data/epgp/manifest_train.jsonl")}
hn = [json.loads(l) for l in open(f"{ROOT}/data/epgp/annot/manifest_hardneg.jsonl")][:8]
for h in hn:
    t = tr.get(h["id"])
    if not t: continue
    vids = glob.glob(f"{ROOT}/data/epgp/wav16k/{t['video']}.wav")
    if not vids: print("no full wav for", t["video"]); continue
    full, sr = sf.read(vids[0], dtype="float32")
    if full.ndim > 1: full = full.mean(1)
    s, d = t["start"], t["dur"]
    cut = full[int(s*16000):int((s+d)*16000)]
    ps = max(0.0, s - PAD)
    pad = full[int(ps*16000):int((s+d)*16000)]
    print(f"\n[{h['id']}] start={s:.1f}s dur={d:.1f}s")
    print(f"  CUT   : {greedy(cut)[:75]}")
    print(f"  +{PAD}s : {greedy(pad)[:75]}")
