#!/bin/bash
cd /home/ece/BigDisk/Prathosh/ASR
P=envs/nemo_ai4b/bin/python
M=data/utts_fa/manifest_fa_eval_sk10.jsonl
Q=data/utts_fa/manifest_prose_eval_vtn.jsonl
# 1) Whisper-medium eval (GPU1) when its FT saves
until grep -q "SAVED exp/whisper_med_v5" logs/whisper_ft.log 2>/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=1 $P scripts/eval_whisper.py --model exp/whisper_med_v5 --manifest $M --tag "whisper-med-ft CHANT" > logs/whisper_med_eval.log 2>&1
CUDA_VISIBLE_DEVICES=1 $P scripts/eval_whisper.py --model exp/whisper_med_v5 --manifest $Q --tag "whisper-med-ft PROSE" >> logs/whisper_med_eval.log 2>&1
echo WMED_EVAL_DONE >> logs/whisper_med_eval.log
# 2) XLSR eval (GPU0) when its FT saves
until grep -q "SAVED exp/w2v2_xlsr_v5" logs/w2v2_ft.log 2>/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 $P scripts/eval_w2v2.py --model exp/w2v2_xlsr_v5 --manifest $M --tag "XLSR-ft CHANT" > logs/w2v2_eval.log 2>&1
CUDA_VISIBLE_DEVICES=0 $P scripts/eval_w2v2.py --model exp/w2v2_xlsr_v5 --manifest $Q --tag "XLSR-ft PROSE" >> logs/w2v2_eval.log 2>&1
echo W2V2_EVAL_DONE >> logs/w2v2_eval.log
# 3) 3-way oracle / complementarity (chant)
CUDA_VISIBLE_DEVICES=0 $P scripts/oracle_ensemble.py --xlsr exp/w2v2_xlsr_v5 > logs/oracle.log 2>&1
echo BAKEOFF_ALL_DONE >> logs/oracle.log
