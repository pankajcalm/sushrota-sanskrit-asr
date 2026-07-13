#!/usr/bin/env python3
"""Confirm CTC head application + sa-slice greedy sanity + frame rate."""
import json, numpy as np, torch, soundfile as sf, re, unicodedata
import nemo.collections.asr as nemo_asr
MODEL=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
       "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
       "indicconformer_stt_sa_hybrid_rnnt_large.nemo")
m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL, map_location="cuda"); m.eval()
print("has ctc_decoder:", hasattr(m, "ctc_decoder"), "| has ctc_decoding:", hasattr(m, "ctc_decoding"))
wav="/home/ece/BigDisk/Prathosh/ASR/data/norm16k/sri_skanda_3_adhyaya_29.wav"
audio, sr = sf.read(wav); clip = audio[:40*sr].astype(np.float32)
sig=torch.tensor(clip).unsqueeze(0).cuda(); sl=torch.tensor([clip.shape[0]]).cuda()
with torch.no_grad():
    enc, enc_len = m.forward(input_signal=sig, input_signal_length=sl)
    lp = m.ctc_decoder(encoder_output=enc)           # expect [B,T,5633]
print("ctc_decoder out shape:", tuple(lp.shape), "enc_len:", int(enc_len[0]))
lp = lp[0].cpu().numpy()                              # [T,5633]
T = lp.shape[0]; print("T frames:", T, "-> index_duration:", round(40.0/T, 4), "s/frame")
# sa slice
off, V, BLANK = 4096, 256, 5632
cols = list(range(off, off+V)) + [BLANK]
sl2 = lp[:, cols]
sl2 = sl2 - sl2.max(-1, keepdims=True); e=np.exp(sl2); sl2 = np.log(e/e.sum(-1, keepdims=True)+1e-12)
sv = json.load(open("/home/ece/BigDisk/Prathosh/ASR/lm/sa_vocab.json"))
labels = sv["tokens"] + [""]
ids = sl2.argmax(-1); out=[]; prev=-1
for i in ids:
    if i != prev and i != len(labels)-1: out.append(labels[i])
    prev=i
txt = "".join(out).replace("▁", " ").strip()
print("greedy(sa-slice, 40s):", txt[:120])
