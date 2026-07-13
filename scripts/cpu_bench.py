#!/usr/bin/env python3
"""CPU inference latency/RTF of ft_ctc_v5 via the raw encoder->CTC forward (the
serving path; bypasses the multilingual transcribe() language-mask wrinkle)."""
import os, time, json
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch, soundfile as sf
torch.set_num_threads(4)
import nemo.collections.asr as nemo_asr
ROOT="/home/ece/BigDisk/Prathosh/ASR"
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cpu")
m.eval()
n=sum(p.numel() for p in m.parameters())
print(f"params: {n/1e6:.1f}M   (fp32 ~{n*4/1e6:.0f}MB, int8 ~{n/1e6:.0f}MB)")
OFF=4096; V=256; BLANK=5632
rows=[json.loads(l) for l in open(f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl")][:8]
def infer(fp):
    x,sr=sf.read(fp, dtype="float32")
    if x.ndim>1: x=x.mean(1)
    sig=torch.tensor(x).unsqueeze(0); sl=torch.tensor([x.shape[0]])
    with torch.no_grad():
        enc,_=m.forward(input_signal=sig, input_signal_length=sl)
        lp=m.ctc_decoder(encoder_output=enc)[0]  # [T, 5633]
    cols=[BLANK]+list(range(OFF,OFF+V))
    ids=lp[:,cols].argmax(-1)
    return ids
_=infer(rows[0]["audio_filepath"])  # warmup
t=time.time()
for r in rows: infer(r["audio_filepath"])
wall=time.time()-t
aud=sum(r["duration"] for r in rows)
print(f"threads={torch.get_num_threads()}  clips={len(rows)}  audio={aud:.1f}s  wall={wall:.2f}s")
print(f"RTF = {wall/aud:.3f}   avg latency = {wall/len(rows)*1000:.0f} ms/clip (avg {aud/len(rows):.1f}s audio)")
