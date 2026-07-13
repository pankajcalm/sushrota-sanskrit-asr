#!/usr/bin/env python3
import json, re, unicodedata, glob, os
import nemo.collections.asr as nemo_asr

MODEL_PATH=('/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/'
 'ai4bharat/indicconformer_stt_sa_hybrid_ctc_rnnt_large/'
 'c82246cc7136e7f1be8df3090e3a07d9/indicconformer_stt_sa_hybrid_rnnt_large.nemo')
CHUNKS=sorted(glob.glob('/tmp/ait_chunks/chunk_*.wav'))
KEEP=re.compile(r'[^ऀ-ॿ\s]')
def norm(s):
    s=unicodedata.normalize('NFC',s).replace('ॐ',' '); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def txt(h):
    if isinstance(h,str): return h
    if hasattr(h,'text'): return h.text
    if isinstance(h,(list,tuple)) and h: return txt(h[0])
    return str(h)
def unwrap(o):
    if isinstance(o,tuple): o=o[0]
    return [txt(h) for h in o]

print('loading ASR...',flush=True)
m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_PATH,map_location='cuda')
m.eval()
try: m.cur_decoder='ctc'
except Exception: pass
m.change_decoding_strategy(decoder_type='ctc')
print(f'transcribing {len(CHUNKS)} chunks (CTC)...',flush=True)
hyps=[norm(h) for h in unwrap(m.transcribe(CHUNKS,batch_size=16,language_id='sa'))]
raw=' '.join(hyps)
print('=== RAW CTC (Devanagari) ===',flush=True); print(raw,flush=True)

# post-correction
import torch
from transformers import AutoTokenizer,T5ForConditionalGeneration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
d2i=lambda s: transliterate(s,sanscript.DEVANAGARI,sanscript.IAST)
i2d=lambda s: transliterate(s,sanscript.IAST,sanscript.DEVANAGARI)
BASE='buddhist-nlp/byt5-sanskrit'; PC='/home/ece/BigDisk/Prathosh/ASR/exp/byt5_postcorr/best'
print('loading post-corrector...',flush=True)
tok=AutoTokenizer.from_pretrained(BASE); pc=T5ForConditionalGeneration.from_pretrained(PC).cuda().eval()
cor=[]
with torch.no_grad():
    for h in hyps:
        if not h: cor.append(''); continue
        enc=tok([d2i(h)],return_tensors='pt',padding=True,truncation=True,max_length=384).to('cuda')
        g=pc.generate(**enc,max_length=384,num_beams=4)
        cor.append(i2d(tok.decode(g[0],skip_special_tokens=True)))
corr=' '.join(cor)
print('=== POST-CORRECTED (Devanagari) ===',flush=True); print(corr,flush=True)
json.dump({'raw':raw,'corrected':corr,'chunks_raw':hyps,'chunks_corr':cor},
    open('/tmp/ait_result.json','w'),ensure_ascii=False,indent=2)
print('SAVED /tmp/ait_result.json',flush=True)
