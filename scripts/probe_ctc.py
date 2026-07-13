#!/usr/bin/env python3
import inspect, torch
from omegaconf import open_dict
import nemo.collections.asr as nemo_asr
ROOT="/home/ece/BigDisk/Prathosh/ASR"
MODEL=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
       "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
       "indicconformer_stt_sa_hybrid_rnnt_large.nemo")
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL, map_location="cuda"); m.train()
with open_dict(m.cfg):
    m.cfg.train_ds.manifest_filepath=[f"{ROOT}/data/train_ft.jsonl"]
    m.cfg.train_ds.is_concat=False; m.cfg.train_ds.batch_size=4
    m.cfg.train_ds.shuffle=False; m.cfg.train_ds.num_workers=0
    m.cfg.train_ds.max_duration=20.0; m.cfg.train_ds.min_duration=1.0
m.setup_training_data(m.cfg.train_ds)
print("ctc_decoder.forward sig:", str(inspect.signature(m.ctc_decoder.forward)))
b=next(iter(m._train_dl))
sig, siglen, tr, trlen, sid, lang = b
sig, siglen, tr, trlen = sig.cuda(), siglen.cuda(), tr.cuda(), trlen.cuda()
lang = lang.cuda() if torch.is_tensor(lang) else lang
print("lang ids sample:", lang[:4] if torch.is_tensor(lang) else lang)
enc, enclen = m.forward(input_signal=sig, input_signal_length=siglen)
print("encoded:", tuple(enc.shape), "enclen:", enclen[:4].tolist())
# try ctc_decoder with / without language_ids
try:
    lp = m.ctc_decoder(encoder_output=enc, language_ids=lang)
    mode="with language_ids"
except TypeError as e:
    lp = m.ctc_decoder(encoder_output=enc); mode="no language_ids"
print("ctc log_probs:", tuple(lp.shape), "|", mode)
loss = m.ctc_loss(log_probs=lp, targets=tr, input_lengths=enclen, target_lengths=trlen)
print("CTC loss:", float(loss))
loss.backward()
print("BACKWARD OK — CTC-only training path works (no numba)")
