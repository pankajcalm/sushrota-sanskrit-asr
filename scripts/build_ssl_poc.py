#!/usr/bin/env python3
"""SSL POC: build a SpeechEncDecSelfSupervisedModel matching v5's conformer, transplant v5's
encoder+preprocessor, run contrastive SSL (loss must DROP), round-trip encoder back to CTC.
De-risks the fork integration before the full ~68h run."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import json
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
import nemo.collections.asr as na
from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
def stage(s): print(f"\n==== {s} ====", flush=True)

stage("1. load v5 for cfg + weights")
src = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cpu")
pre_cfg = OmegaConf.to_container(src.cfg.preprocessor, resolve=True)
enc_cfg = OmegaConf.to_container(src.cfg.encoder, resolve=True)
d_model = enc_cfg["d_model"]; feats = pre_cfg.get("features", 80)
pre_cfg["pad_to"] = 16   # round batch time dim to a multiple of combine_time_steps(4) for the contrastive reshape
print(f"d_model={d_model} feats={feats} pad_to={pre_cfg['pad_to']}", flush=True)

stage("2. smoke SSL manifest (audio-only)")
SSLMAN = f"{ROOT}/data/epgp/ssl_smoke.jsonl"
rows = [json.loads(l) for l in open(f"{ROOT}/data/epgp/manifest_train.jsonl")]
rows = [r for r in rows if 2.0 <= r.get("dur", 0) <= 18.0][:320]
with open(SSLMAN, "w") as f:
    for r in rows: f.write(json.dumps({"audio_filepath": r["audio_filepath"], "duration": r["dur"], "text": ""}) + "\n")
print(f"{len(rows)} clips", flush=True)

stage("3. build SSL cfg")
cfg = OmegaConf.create({
    "sample_rate": 16000,
    "train_ds": {"manifest_filepath": SSLMAN, "sample_rate": 16000, "batch_size": 8,
                 "shuffle": True, "num_workers": 4, "min_duration": 2.0, "max_duration": 18.0},
    "preprocessor": pre_cfg,
    "spec_augment": {"_target_": "nemo.collections.asr.modules.MaskedPatchAugmentation",
                     "patch_size": 48, "mask_patches": 0.5, "freq_masks": 3, "freq_width": 20},
    "encoder": enc_cfg,
    "loss_list": {"contrastive": {
        "decoder": {"_target_": "nemo.collections.asr.modules.ConvASRDecoderReconstruction",
                    "feat_in": d_model, "feat_hidden": 128, "feat_out": 128,
                    "stride_layers": 0, "non_stride_layers": 0},
        "loss": {"_target_": "nemo.collections.asr.losses.ContrastiveLoss",
                 "in_dim": feats, "proj_dim": 128, "combine_time_steps": 4,
                 "quantized_targets": True, "codebook_size": 300, "num_negatives": 50,
                 "sample_from_same_utterance_only": False, "sample_from_non_masked": False}}},
    "optim": {"name": "adamw", "lr": 3e-4, "weight_decay": 1e-3},
})

stage("4. instantiate SSL model + trainer")
class LossRec(pl.Callback):
    def __init__(self): self.losses = []
    def on_train_batch_end(self, tr, plm, outputs, batch, bidx):
        l = outputs["loss"] if isinstance(outputs, dict) else outputs
        if l is not None: self.losses.append(float(l))
rec = LossRec()
trainer = pl.Trainer(accelerator="gpu", devices=1, max_steps=40, logger=False,
                     enable_checkpointing=False, enable_progress_bar=False,
                     num_sanity_val_steps=0, log_every_n_steps=5, callbacks=[rec])
model = SpeechEncDecSelfSupervisedModel(cfg=cfg, trainer=trainer)
print("instantiated with trainer", flush=True)

stage("5. transplant v5 encoder + preprocessor weights")
sd = src.state_dict()
graft = {k: v for k, v in sd.items() if k.startswith("encoder.") or k.startswith("preprocessor.")}
missing, unexpected = model.load_state_dict(graft, strict=False)
print(f"grafted {len(graft)} | encoder-missing {len([m for m in missing if m.startswith('encoder.')])} | unexpected {len(unexpected)}", flush=True)

stage("6. SSL via trainer.fit — loss must DECREASE")
trainer.fit(model)
ls = rec.losses
if len(ls) >= 10:
    f5, l5 = sum(ls[:5]) / 5, sum(ls[-5:]) / 5
    print(f"steps={len(ls)}  first5 {f5:.3f}  last5 {l5:.3f}  DELTA {f5 - l5:+.3f}", flush=True)
else:
    print(f"steps={len(ls)} losses={[round(x,2) for x in ls]}", flush=True)

stage("7. round-trip: adapted encoder -> CTC, greedy not broken")
new_enc = {k: v.cpu() for k, v in model.state_dict().items() if k.startswith("encoder.")}
src.load_state_dict(new_enc, strict=False)
import numpy as np, soundfile as sf
LAB = json.load(open(f"{ROOT}/data/eval_logits/labels.json")); OFF, VV, BL = 4096, 256, 5632
def lse(a, ax): m = a.max(ax, keepdims=True); return m + np.log(np.exp(a - m).sum(ax, keepdims=True))
src = src.to("cuda").eval()
wav, _ = sf.read(rows[0]["audio_filepath"], dtype="float32")
sig = torch.tensor(wav).unsqueeze(0).cuda(); sl = torch.tensor([len(wav)]).cuda()
with torch.no_grad():
    enc, _ = src.forward(input_signal=sig, input_signal_length=sl)
    lp = src.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
cols = [BL] + list(range(OFF, OFF + VV)); P = lp[:, cols]; P = P - lse(P, 1); ids = P.argmax(1)
out = []; prev = -1
for i in ids:
    i = int(i)
    if i != prev and i != 0: out.append(LAB[i - 1])
    prev = i
print("greedy(adapted enc):", ''.join(out).replace('▁', ' ').strip()[:80], flush=True)
print("\nPOC DONE", flush=True)
