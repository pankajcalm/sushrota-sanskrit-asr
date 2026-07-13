#!/bin/bash
# Wait for v8 ep10 checkpoint, then run the full CER/WER/SN-WER matrix: v5 vs v8-ep10
# across chant / prose / e-PG-clean / e-PG-phone. Self-contained, tmux-durable.
cd /home/ece/BigDisk/Prathosh/ASR
CK=exp/ft_ctc_v8/ft_ctc_ep10.nemo
echo "[matrix] waiting for $CK ..."
until [ -f "$CK" ]; do sleep 60; done
sleep 30   # let save_to finish flushing
echo "[matrix] ep10 ready. running..."
NAMES=(chant prose epg_clean epg_phone)
MANI=(data/utts_fa/manifest_fa_eval_sk10.jsonl data/utts_fa/manifest_prose_eval_vtn.jsonl data/epgp/eval_gold_clean.jsonl data/epgp/eval_gold_phone.jsonl)
MODELS=("v5:exp/ft_ctc_v5/ft_ctc_ep20.nemo" "v8ep10:exp/ft_ctc_v8/ft_ctc_ep10.nemo")
for i in "${!NAMES[@]}"; do
  for tm in "${MODELS[@]}"; do
    tag="${tm%%:*}"; model="${tm#*:}"
    CUDA_VISIBLE_DEVICES=0 envs/nemo_ai4b/bin/python scripts/eval_model.py --model "$model" --manifest "${MANI[$i]}" --tag "${tag}-${NAMES[$i]}" 2>>logs/matrix_v8.err
  done
done
echo "MATRIX DONE"
