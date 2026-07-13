#!/usr/bin/env python3
"""Prepare NeMo finetune manifests: add lang='sa', split speaker-disjoint dev for early-stop.
Also dump the model's train_ds / optim / spec_augment config so we wire finetune correctly."""
import json, sys, argparse
ROOT="/home/ece/BigDisk/Prathosh/ASR"
_ap=argparse.ArgumentParser()
_ap.add_argument("--in", dest="inp", default=f"{ROOT}/data/train_manifest_v1.jsonl")
_ap.add_argument("--tag", default="")   # "" -> train_ft.jsonl ; "2" -> train_ft2.jsonl
_ap.add_argument("cfg", nargs="?")
_A,_=_ap.parse_known_args()
rows=[json.loads(l) for l in open(_A.inp)]
DEV_SPK={"Srikanth","Ravi","Vishnu","Madhava"}
tr, dv = [], []
for r in rows:
    rec={"audio_filepath":r["audio_filepath"],"text":r["text"],"duration":r["duration"],"lang":"sa"}
    (dv if r.get("speaker") in DEV_SPK else tr).append(rec)
for name, data in [(f"train_ft{_A.tag}", tr), (f"dev_ft{_A.tag}", dv)]:
    with open(f"{ROOT}/data/{name}.jsonl","w") as f:
        for r in data: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"{name}: {len(data)} utts, {sum(r['duration'] for r in data)/3600:.2f}h")

# --- dump model training-relevant config ---
if _A.cfg=="cfg":
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
    MODEL=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
           "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
           "indicconformer_stt_sa_hybrid_rnnt_large.nemo")
    m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL, map_location="cpu")
    c=m.cfg
    print("\n=== train_ds keys ==="); print(OmegaConf.to_yaml(c.train_ds) if 'train_ds' in c else "none")
    print("=== optim ==="); print(OmegaConf.to_yaml(c.optim) if 'optim' in c else "none")
    print("=== spec_augment ==="); print(OmegaConf.to_yaml(c.spec_augment) if 'spec_augment' in c else "none")
    print("=== aux_ctc/decoder present ==="); print("aux_ctc" in c, "| loss_ratio:", c.get("aux_ctc",{}).get("ctc_loss_weight","?"))
