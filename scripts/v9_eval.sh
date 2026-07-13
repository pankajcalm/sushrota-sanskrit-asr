#!/bin/bash
# Wait for v9a + v9b finetunes, then eval v5 vs v9a vs v9b (ep8/ep12) on
# heldout (de-anchored, unseen e-PG speakers) + chant + prose. CER/WER/SN-WER via eval_model.py.
cd /home/ece/BigDisk/Prathosh/ASR
echo "[v9eval] waiting for both finetunes..."
until [ -f exp/ft_ctc_v9a/ft_ctc_final.nemo ] && [ -f exp/ft_ctc_v9b/ft_ctc_final.nemo ]; do sleep 30; done
sleep 20
MODELS=("v5:exp/ft_ctc_v5/ft_ctc_ep20.nemo"
        "v9a-ep8:exp/ft_ctc_v9a/ft_ctc_ep8.nemo"
        "v9a-ep12:exp/ft_ctc_v9a/ft_ctc_ep12.nemo"
        "v9b-ep8:exp/ft_ctc_v9b/ft_ctc_ep8.nemo"
        "v9b-ep12:exp/ft_ctc_v9b/ft_ctc_ep12.nemo")
SN=(heldout chant prose)
SM=(data/epgp/v9_heldout.jsonl data/utts_fa/manifest_fa_eval_sk10.jsonl data/utts_fa/manifest_prose_eval_vtn.jsonl)
for tm in "${MODELS[@]}"; do
  tag="${tm%%:*}"; model="${tm#*:}"
  for i in "${!SN[@]}"; do
    CUDA_VISIBLE_DEVICES=0 envs/nemo_ai4b/bin/python scripts/eval_model.py --model "$model" --manifest "${SM[$i]}" --tag "${tag}|${SN[$i]}" 2>>logs/v9_eval.err
  done
done
echo "V9 EVAL DONE"
