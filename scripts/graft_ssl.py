#!/usr/bin/env python3
"""Graft the SSL-adapted encoder into the base IndicConformer -> base_ssl.nemo for v9b finetune."""
import sys, torch
import nemo.collections.asr as na
ROOT = "/home/ece/BigDisk/Prathosh/ASR"
BASE = ("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
        "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
        "indicconformer_stt_sa_hybrid_rnnt_large.nemo")
ENC = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/exp/ssl_epgp/encoder_final.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/exp/ssl_epgp/base_ssl.nemo"
m = na.models.EncDecHybridRNNTCTCBPEModel.restore_from(BASE, map_location="cpu")
enc = torch.load(ENC, map_location="cpu")
miss, unexp = m.load_state_dict(enc, strict=False)
print(f"grafted {len(enc)} enc tensors | encoder-missing {len([x for x in miss if x.startswith('encoder.')])} | unexpected {len(unexp)}")
m.save_to(OUT); print(f"SAVED {OUT}")
