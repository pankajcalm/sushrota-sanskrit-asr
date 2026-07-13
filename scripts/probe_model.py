#!/usr/bin/env python3
"""Probe: (1) CTC log-prob extraction path + frame rate, (2) sa tokenization."""
import json, numpy as np, torch, soundfile as sf
import nemo.collections.asr as nemo_asr

MODEL=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
       "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
       "indicconformer_stt_sa_hybrid_rnnt_large.nemo")
m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL, map_location="cuda")
m.eval()
try: m.cur_decoder = "ctc"
except Exception as e: print("cur_decoder set err", e)

# ---- tokenizer inspection ----
print("=== tokenizer ===")
print("type:", type(m.tokenizer))
for attr in ("langs", "tokenizers_dict", "vocab_size"):
    print(" ", attr, getattr(m.tokenizer, attr, "N/A") if attr != "tokenizers_dict"
          else list(getattr(m.tokenizer, attr, {}).keys()))
sample = "योगस्य लक्षणं वक्ष्ये सबीजस्य नृपात्मजे"
for fn in ("text_to_ids",):
    try:
        ids = m.tokenizer.text_to_ids(sample, "sa")
        print(f"  text_to_ids(...,'sa') -> len {len(ids)} range[{min(ids)},{max(ids)}] head {ids[:12]}")
    except Exception as e:
        print("  text_to_ids(lang) err:", e)
    try:
        ids = m.tokenizer.text_to_ids(sample)
        print(f"  text_to_ids(no lang) -> len {len(ids)} range[{min(ids)},{max(ids)}] head {ids[:12]}")
    except Exception as e:
        print("  text_to_ids(no lang) err:", e)

# ---- ctc logprob path ----
print("\n=== ctc logprobs ===")
wav = "/home/ece/BigDisk/Prathosh/ASR/data/norm16k/sri_skanda_3_adhyaya_29.wav"
audio, sr = sf.read(wav); audio = audio[:40*sr].astype(np.float32)  # 40s clip
sig = torch.tensor(audio).unsqueeze(0).cuda()
sl = torch.tensor([audio.shape[0]]).cuda()
with torch.no_grad():
    out = m.forward(input_signal=sig, input_signal_length=sl)
print("forward returns", type(out), "len", len(out) if isinstance(out, tuple) else "-")
for i, o in enumerate(out if isinstance(out, tuple) else [out]):
    print(f"  [{i}] {type(o).__name__} shape={getattr(o,'shape',None)}")
