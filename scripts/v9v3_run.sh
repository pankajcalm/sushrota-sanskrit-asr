#!/bin/bash
# v9-v3: formalized split. Train = train_ft5 + ALL annotations (+phone). Gold held out as eval.
# Eval v5 + v9-v3 on the 3-domain suite: gold(e-PG) / chant(Bhagavata) / prose(Vedanta). GPU1.
set -e
cd /home/ece/BigDisk/Prathosh/ASR
echo "[v9v3] prep"
envs/nemo_ai4b/bin/python scripts/v9_prep_v3.py
echo "[v9v3] phone-aug"
envs/nemo_ai4b/bin/python scripts/phone_aug2.py --inmanifest data/epgp/v9v3_clean.jsonl --outdir data/epgp/v9v3_phone --outmanifest data/epgp/v9v3_phone.jsonl
cat data/train_ft5.jsonl data/epgp/v9v3_clean.jsonl data/epgp/v9v3_phone.jsonl > data/v9v3_train.jsonl
echo "[v9v3] train rows: $(wc -l < data/v9v3_train.jsonl)"
CUDA_VISIBLE_DEVICES=1 envs/nemo_ai4b/bin/python scripts/finetune_ctc_init.py \
  --train data/v9v3_train.jsonl --outdir exp/ft_ctc_v9v3 --epochs 20 --save-eps 10,15,20 --bs 16 --lr 1e-4
echo "[v9v3] eval"
SN=(gold chant prose)
SM=(data/epgp/eval_gold.jsonl data/utts_fa/manifest_fa_eval_sk10.jsonl data/utts_fa/manifest_prose_eval_vtn.jsonl)
for i in "${!SN[@]}"; do
  CUDA_VISIBLE_DEVICES=1 envs/nemo_ai4b/bin/python scripts/eval_model.py --model exp/ft_ctc_v5/ft_ctc_ep20.nemo --manifest "${SM[$i]}" --tag "v5|${SN[$i]}" 2>>logs/v9v3.err
done
for ep in 15 20; do
  for i in "${!SN[@]}"; do
    CUDA_VISIBLE_DEVICES=1 envs/nemo_ai4b/bin/python scripts/eval_model.py --model exp/ft_ctc_v9v3/ft_ctc_ep${ep}.nemo --manifest "${SM[$i]}" --tag "v9v3-ep${ep}|${SN[$i]}" 2>>logs/v9v3.err
  done
done
echo "V9V3 ALL DONE"
