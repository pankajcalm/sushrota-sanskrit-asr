#!/usr/bin/env python3
"""CTC-only adaptation of IndicConformer hybrid (RNNT head bypassed — numba-incompatible
and unused since we decode CTC). LOW LR to protect the SSL backbone. Saves .nemo at
checkpoint epochs so each can be evaluated on the real sk10 held-out set (overfitting curve)."""
import argparse, types, os
import torch, pytorch_lightning as pl
from omegaconf import open_dict
import nemo.collections.asr as nemo_asr

ROOT="/home/ece/BigDisk/Prathosh/ASR"
MODEL=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
       "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
       "indicconformer_stt_sa_hybrid_rnnt_large.nemo")

def ctc_training_step(self, batch, batch_nb):
    signal, signal_len, transcript, transcript_len, sample_ids, language_ids = batch
    encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
    log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids)
    loss = self.ctc_loss(log_probs=log_probs, targets=transcript,
                         input_lengths=encoded_len, target_lengths=transcript_len)
    self.log("train_loss", loss, prog_bar=True, on_step=True)
    self.log("lr", self._optimizer.param_groups[0]["lr"], prog_bar=True, on_step=True)
    return loss

class SaveNemo(pl.Callback):
    def __init__(self, epochs, outdir): self.epochs=set(epochs); self.outdir=outdir
    def on_train_epoch_end(self, trainer, pl_module):
        e=trainer.current_epoch+1
        if e in self.epochs:
            p=f"{self.outdir}/ft_ctc_ep{e}.nemo"; pl_module.save_to(p)
            print(f"[ckpt] saved {p}", flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--save-eps", default="3,6,10")
    ap.add_argument("--outdir", default=f"{ROOT}/exp/ft_ctc_v1")
    ap.add_argument("--train", default=f"{ROOT}/data/train_ft.jsonl")
    a=ap.parse_args()
    outdir=a.outdir; os.makedirs(outdir, exist_ok=True)

    m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL, map_location="cuda")
    with open_dict(m.cfg):
        m.cfg.train_ds.manifest_filepath=[a.train]
        m.cfg.train_ds.is_concat=False; m.cfg.train_ds.batch_size=a.bs
        m.cfg.train_ds.shuffle=True; m.cfg.train_ds.num_workers=8
        m.cfg.train_ds.max_duration=20.0; m.cfg.train_ds.min_duration=1.0
        m.cfg.optim.name="adamw"; m.cfg.optim.lr=a.lr; m.cfg.optim.weight_decay=1e-3
        m.cfg.optim.sched.name="CosineAnnealing"
        m.cfg.optim.sched.warmup_steps=50 if a.smoke else 500
        m.cfg.optim.sched.min_lr=1e-6
        if "d_model" in m.cfg.optim.sched: del m.cfg.optim.sched.d_model
    m.setup_training_data(m.cfg.train_ds)
    m.training_step=types.MethodType(ctc_training_step, m)   # CTC-only, bypass RNNT numba

    cbs=[] if a.smoke else [SaveNemo([int(x) for x in a.save_eps.split(",")], outdir)]
    trainer=pl.Trainer(
        accelerator="gpu", devices=1,
        max_epochs=1 if a.smoke else a.epochs,
        max_steps=10 if a.smoke else -1,
        limit_train_batches=10 if a.smoke else 1.0,
        limit_val_batches=0.0, num_sanity_val_steps=0,   # no in-loop val
        precision=(32 if a.smoke else "bf16"),
        logger=False, enable_checkpointing=False, callbacks=cbs,
        gradient_clip_val=1.0, log_every_n_steps=20, default_root_dir=outdir,
    )
    m.set_trainer(trainer)
    print(f"[finetune] smoke={a.smoke} lr={a.lr} bs={a.bs} epochs={a.epochs}", flush=True)
    trainer.fit(m)
    if not a.smoke:
        m.save_to(f"{outdir}/ft_ctc_final.nemo"); print(f"SAVED {outdir}/ft_ctc_final.nemo", flush=True)
    print("DONE", flush=True)

if __name__=="__main__": main()
