#!/usr/bin/env python3
"""Full SSL continued-pretraining: adapt v5's conformer encoder to ~40h of e-PG śāstric
Sanskrit audio (unlabeled) via wav2vec-style contrastive SSL. Saves the adapted encoder for
the v9 CTC finetune. Harness validated by build_ssl_poc.py."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import json
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
import nemo.collections.asr as na
from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
OUT = f"{ROOT}/exp/ssl_epgp"; os.makedirs(OUT, exist_ok=True)
EPOCHS = 15; BS = 16
def stage(s): print(f"\n==== {s} ====", flush=True)

stage("cfg + weights from v5")
src = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cpu")
pre_cfg = OmegaConf.to_container(src.cfg.preprocessor, resolve=True)
enc_cfg = OmegaConf.to_container(src.cfg.encoder, resolve=True)
pre_cfg["pad_to"] = 16
d_model = enc_cfg["d_model"]; feats = pre_cfg.get("features", 80)

stage("SSL manifest (all e-PG segments, audio-only)")
SSLMAN = f"{ROOT}/data/epgp/ssl_train.jsonl"
rows = [json.loads(l) for l in open(f"{ROOT}/data/epgp/manifest_train.jsonl")]
rows = [r for r in rows if 2.0 <= r.get("dur", 0) <= 18.0]
hrs = sum(r["dur"] for r in rows) / 3600
with open(SSLMAN, "w") as f:
    for r in rows: f.write(json.dumps({"audio_filepath": r["audio_filepath"], "duration": r["dur"], "text": ""}) + "\n")
print(f"{len(rows)} clips, {hrs:.1f}h", flush=True)
steps_per_epoch = (len(rows) + BS - 1) // BS
max_steps = steps_per_epoch * EPOCHS
print(f"steps/epoch {steps_per_epoch}  max_steps {max_steps}", flush=True)

stage("SSL cfg")
cfg = OmegaConf.create({
    "sample_rate": 16000,
    "train_ds": {"manifest_filepath": SSLMAN, "sample_rate": 16000, "batch_size": BS,
                 "shuffle": True, "num_workers": 8, "min_duration": 2.0, "max_duration": 18.0, "pin_memory": True},
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
    "optim": {"name": "adamw", "lr": 3e-4, "weight_decay": 1e-3,
              "sched": {"name": "CosineAnnealing", "warmup_steps": 1000, "min_lr": 1e-5, "max_steps": max_steps}},
})

stage("build + transplant")
class Log(pl.Callback):
    def on_train_batch_end(self, tr, plm, outputs, batch, bidx):
        if tr.global_step % 200 == 0:
            l = outputs["loss"] if isinstance(outputs, dict) else outputs
            print(f"  step {tr.global_step}/{max_steps}  loss {float(l):.1f}", flush=True)
    def on_train_epoch_end(self, tr, plm):
        e = tr.current_epoch
        if (e + 1) % 5 == 0 or (e + 1) == EPOCHS:
            enc = {k: v.cpu() for k, v in plm.state_dict().items() if k.startswith("encoder.")}
            torch.save(enc, f"{OUT}/encoder_ep{e+1}.pt"); print(f"[save] encoder_ep{e+1}.pt", flush=True)
trainer = pl.Trainer(accelerator="gpu", devices=1, max_steps=max_steps, max_epochs=EPOCHS,
                     precision="bf16", logger=False, enable_checkpointing=False,
                     num_sanity_val_steps=0, log_every_n_steps=200, callbacks=[Log()])
model = SpeechEncDecSelfSupervisedModel(cfg=cfg, trainer=trainer)
graft = {k: v for k, v in src.state_dict().items() if k.startswith("encoder.") or k.startswith("preprocessor.")}
miss, unexp = model.load_state_dict(graft, strict=False)
print(f"grafted {len(graft)} | enc-missing {len([m for m in miss if m.startswith('encoder.')])} | unexpected {len(unexp)}", flush=True)

# FREEZE conv frontend + bottom 4 of 17 conformer layers; adapt the top 13 (+ SSL head).
# Low-level acoustics are language-universal & channel-coupled -> keep them; śāstric structure lives up top.
FREEZE_LAYERS = 4
for p in model.encoder.pre_encode.parameters(): p.requires_grad = False
for i in range(FREEZE_LAYERS):
    for p in model.encoder.layers[i].parameters(): p.requires_grad = False
ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
ntot = sum(p.numel() for p in model.parameters())
print(f"frozen frontend + layers 0..{FREEZE_LAYERS-1}; trainable {ntr/1e6:.1f}M / {ntot/1e6:.1f}M", flush=True)

stage(f"train SSL — {EPOCHS} epochs, {hrs:.0f}h audio")
trainer.fit(model)
enc = {k: v.cpu() for k, v in model.state_dict().items() if k.startswith("encoder.")}
torch.save(enc, f"{OUT}/encoder_final.pt")
print(f"\nSSL DONE -> {OUT}/encoder_final.pt", flush=True)
