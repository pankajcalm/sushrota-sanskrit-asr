#!/usr/bin/env python3
import inspect, json
from omegaconf import open_dict
import nemo.collections.asr as nemo_asr
ROOT="/home/ece/BigDisk/Prathosh/ASR"
MODEL=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
       "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
       "indicconformer_stt_sa_hybrid_rnnt_large.nemo")
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL, map_location="cuda")
with open_dict(m.cfg):
    m.cfg.train_ds.manifest_filepath=[f"{ROOT}/data/train_ft.jsonl"]
    m.cfg.train_ds.is_concat=False; m.cfg.train_ds.batch_size=4
    m.cfg.train_ds.shuffle=False; m.cfg.train_ds.num_workers=0
    m.cfg.train_ds.max_duration=20.0; m.cfg.train_ds.min_duration=1.0
m.setup_training_data(m.cfg.train_ds)
batch=next(iter(m._train_dl))
print("=== BATCH ===")
print("len:", len(batch))
for i,x in enumerate(batch):
    import torch
    print(f"  [{i}] {type(x).__name__}", tuple(x.shape) if torch.is_tensor(x) else x)
print("\n=== components ===")
for a in ("ctc_decoder","ctc_loss","loss","decoder","joint","encoder"):
    print(f"  {a}: {hasattr(m,a)}")
print("\n=== training_step source ===")
print(inspect.getsource(m.training_step)[:2600])
