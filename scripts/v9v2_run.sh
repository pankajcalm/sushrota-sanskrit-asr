#!/bin/bash
# v9-v2: cut on the current 859 annotations + 305 gold, hold out 5 speakers (bigger de-anchored
# eval), from base, 20 epochs. Then eval v9-v2 + v5 on heldout/chant/prose. Fully unattended.
set -e
cd /home/ece/BigDisk/Prathosh/ASR
echo "[v9v2] prep (snapshot of annotations)"
envs/nemo_ai4b/bin/python scripts/v9_prep.py
echo "[v9v2] phone-aug"
envs/nemo_ai4b/bin/python scripts/phone_aug2.py --inmanifest data/epgp/v9_clean.jsonl --outdir data/epgp/v9_phone --outmanifest data/epgp/v9_phone.jsonl
cat data/train_ft5.jsonl data/epgp/v9_clean.jsonl data/epgp/v9_phone.jsonl > data/v9v2_train.jsonl
echo "[v9v2] train rows: $(wc -l < data/v9v2_train.jsonl) | heldout: $(wc -l < data/epgp/v9_heldout.jsonl)"
CUDA_VISIBLE_DEVICES=0 envs/nemo_ai4b/bin/python scripts/finetune_ctc_init.py \
  --train data/v9v2_train.jsonl --outdir exp/ft_ctc_v9v2 --epochs 20 --save-eps 10,15,20 --bs 16 --lr 1e-4
echo "[v9v2] eval"
SN=(heldout chant prose)
SM=(data/epgp/v9_heldout.jsonl data/utts_fa/manifest_fa_eval_sk10.jsonl data/utts_fa/manifest_prose_eval_vtn.jsonl)
# v5 baseline on the NEW (bigger) heldout + reuse known chant/prose
CUDA_VISIBLE_DEVICES=0 envs/nemo_ai4b/bin/python scripts/eval_model.py --model exp/ft_ctc_v5/ft_ctc_ep20.nemo --manifest data/epgp/v9_heldout.jsonl --tag "v5|heldout" 2>>logs/v9v2.err
for ep in 15 20; do
  for i in "${!SN[@]}"; do
    CUDA_VISIBLE_DEVICES=0 envs/nemo_ai4b/bin/python scripts/eval_model.py \
      --model exp/ft_ctc_v9v2/ft_ctc_ep${ep}.nemo --manifest "${SM[$i]}" --tag "v9v2-ep${ep}|${SN[$i]}" 2>>logs/v9v2.err
  done
done
echo "V9V2 ALL DONE"
