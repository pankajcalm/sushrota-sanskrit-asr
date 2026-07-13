#!/bin/bash
# Wait for pseudo-labels, assemble v9pl train (= v9a's set + high-conf-agreed pseudo),
# finetune from base, eval v9pl vs (already-known) v9a/v5 on heldout+chant+prose.
cd /home/ece/BigDisk/Prathosh/ASR
echo "[v9pl] waiting for pseudo-labels..."
until grep -q "PSEUDO DONE" logs/pseudo.log 2>/dev/null; do sleep 30; done
sleep 5
cat data/train_ft5.jsonl data/epgp/v9_clean.jsonl data/epgp/v9_phone.jsonl data/epgp/pseudo.jsonl > data/v9pl_train.jsonl
echo "[v9pl] train manifest: $(wc -l < data/v9pl_train.jsonl) rows (pseudo: $(wc -l < data/epgp/pseudo.jsonl))"
CUDA_VISIBLE_DEVICES=0 envs/nemo_ai4b/bin/python scripts/finetune_ctc_init.py \
  --train data/v9pl_train.jsonl --outdir exp/ft_ctc_v9pl --epochs 12 --save-eps 8,12 --bs 16 --lr 1e-4
echo "[v9pl] eval"
SN=(heldout chant prose)
SM=(data/epgp/v9_heldout.jsonl data/utts_fa/manifest_fa_eval_sk10.jsonl data/utts_fa/manifest_prose_eval_vtn.jsonl)
for ep in 8 12; do
  for i in "${!SN[@]}"; do
    CUDA_VISIBLE_DEVICES=0 envs/nemo_ai4b/bin/python scripts/eval_model.py \
      --model exp/ft_ctc_v9pl/ft_ctc_ep${ep}.nemo --manifest "${SM[$i]}" --tag "v9pl-ep${ep}|${SN[$i]}" 2>>logs/v9pl.err
  done
done
echo "V9PL ALL DONE"
