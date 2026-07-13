#!/usr/bin/env python3
"""Export sa-slice CTC logprobs [T,257] per eval utt from a given model (for char-LM rescoring)."""
import os, json, re, unicodedata, argparse
import numpy as np, torch, soundfile as sf
import nemo.collections.asr as nemo_asr
ROOT="/home/ece/BigDisk/Prathosh/ASR"
MANIFEST=f"{ROOT}/data/utts_fa/manifest_fa_eval_sk10.jsonl"
OUT=f"{ROOT}/data/eval_logits"; os.makedirs(OUT, exist_ok=True)
OFF,V,BLANK=4096,256,5632; COLS=list(range(OFF,OFF+V))+[BLANK]   # sa tokens then blank (blank last)
KEEP=re.compile(r'[^ऀ-ॿ\s]'); DROP=re.compile(r'[०-९।॥ऽॐ]')
def norm(s):
    s=unicodedata.normalize('NFC',s); s=DROP.sub(' ',s); s=KEEP.sub(' ',s)
    return re.sub(r'\s+',' ',s).strip()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model", required=True); a=ap.parse_args()
    man=[json.loads(l) for l in open(MANIFEST)]
    m=nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(a.model, map_location="cuda"); m.eval()
    refs=[]; B=16
    for s in range(0, len(man), B):
        chunk=man[s:s+B]; wavs=[sf.read(u["audio_filepath"])[0].astype(np.float32) for u in chunk]
        L=[len(w) for w in wavs]; mx=max(L)
        sig=torch.zeros(len(wavs), mx)
        for i,w in enumerate(wavs): sig[i,:len(w)]=torch.tensor(w)
        sig=sig.cuda(); sl=torch.tensor(L).cuda()
        with torch.no_grad():
            enc,enclen=m.forward(input_signal=sig, input_signal_length=sl)
            lp=m.ctc_decoder(encoder_output=enc).cpu().numpy()          # [B,T,5633]
        for i,u in enumerate(chunk):
            T=int(enclen[i]); x=lp[i,:T][:,COLS]
            x=x-x.max(-1,keepdims=True); e=np.exp(x); x=np.log(e/e.sum(-1,keepdims=True)+1e-12)
            np.save(f"{OUT}/{s+i:04d}.npy", x.astype(np.float32)); refs.append(norm(u["text"]))
        print(f"  exported {s+len(chunk)}/{len(man)}", flush=True)
    json.dump(refs, open(f"{OUT}/refs.json","w"), ensure_ascii=False)
    sv=json.load(open(f"{ROOT}/lm/sa_vocab.json")); json.dump(sv["tokens"]+[""], open(f"{OUT}/labels.json","w"), ensure_ascii=False)
    print(f"DONE {len(man)} logits -> {OUT}", flush=True)
if __name__=="__main__": main()
