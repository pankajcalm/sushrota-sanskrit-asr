#!/usr/bin/env python3
"""Hard-negative selection v2: rank by v5-vs-Whisper DISAGREEMENT (a real 'one of them is
wrong here' signal) instead of OOV (which pulled correct-but-rare words). Two stages so
Whisper (slow) only runs on a v5-shortlisted pool.
 Stage1: v5 over the full pool -> draft + entropy (cached to v5_pool.jsonl). Shortlist top-N
         by entropy per speaker.
 Stage2: Whisper-ft over the shortlist.
 Stage3: disagreement = normalized word-level edit distance; keep the BAND [0.25,0.9]
         (drops agreement=v5-correct AND total-garbage=English/noise), rank desc, 20/speaker
         with temporal spread. Prefill = v5 (the stronger model)."""
import os, json, re, shutil, unicodedata
from collections import defaultdict
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as nemo_asr
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
OFF, V, BLANK = 4096, 256, 5632
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
POOL = f"{ROOT}/data/epgp/manifest_train.jsonl"
ADIR = f"{ROOT}/data/epgp/annot"; CLIPS = f"{ROOT}/data/epgp/annot_clips"
os.makedirs(CLIPS, exist_ok=True)
CACHE = f"{ADIR}/v5_pool.jsonl"; OUTMAN = f"{ADIR}/manifest_hardneg.jsonl"
WCK = f"{ROOT}/exp/whisper_med_v5/checkpoint-3290"
SHORT, PER_SPK, MINGAP, DMIN, DMAX = 50, 20, 10.0, 3.0, 12.0
DIS_LO, DIS_HI = 0.25, 0.90

KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def ed(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    p = list(range(m + 1))
    for i in range(1, n + 1):
        c = [i] + [0] * m; ai = a[i - 1]
        for j in range(1, m + 1): c[j] = min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ai != b[j - 1]))
        p = c
    return p[m]
def lse(a, ax): mx = a.max(ax, keepdims=True); return mx + np.log(np.exp(a - mx).sum(ax, keepdims=True))
def loadwav(fp):
    x, sr = sf.read(fp, dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    return x

# ---------- Stage 1: v5 pool (cached) ----------
if os.path.exists(CACHE):
    pool = [json.loads(l) for l in open(CACHE, encoding="utf-8")]
    print(f"[stage1] loaded cache {len(pool)} clips", flush=True)
else:
    print(f"[stage1] v5 on {DEV}", flush=True)
    M = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location=DEV); M.eval()
    LABELS = json.load(open(f"{ROOT}/data/eval_logits/labels.json")); LOGN = np.log(V + 1)
    def v5_analyze(wav):
        sig = torch.tensor(wav).unsqueeze(0).to(DEV); sl = torch.tensor([len(wav)]).to(DEV)
        with torch.no_grad():
            enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
            lp = M.ctc_decoder(encoder_output=enc)[0].detach().cpu().numpy()
        cols = [BLANK] + list(range(OFF, OFF + V)); P = lp[:, cols]; P = P - lse(P, 1)
        ids = P.argmax(1); probs = np.exp(P); ent = -(probs * P).sum(1) / LOGN
        emit = ids != 0; ment = float(ent[emit].mean()) if emit.any() else 0.0
        out = []; prev = -1
        for i in ids:
            i = int(i)
            if i != prev and i != 0: out.append(LABELS[i - 1])
            prev = i
        return ''.join(out).replace('▁', ' ').strip(), ment
    rows = [json.loads(l) for l in open(POOL)]
    rows = [r for r in rows if DMIN <= r.get("dur", 0) <= DMAX]
    print(f"[stage1] {len(rows)} pool clips", flush=True)
    pool = []
    with open(CACHE, "w", encoding="utf-8") as cf:
        for k, r in enumerate(rows):
            try: wav = loadwav(r["audio_filepath"])
            except Exception: continue
            if len(wav) == 0 or float(np.sqrt(np.mean(wav ** 2))) < 0.006: continue
            draft, ent = v5_analyze(wav); nw = len(norm(draft).split())
            if nw < 3: continue
            row = {"audio_filepath": r["audio_filepath"], "video": r["video"],
                   "start": r.get("start", 0.0), "dur": r["dur"], "draft": draft, "ent": round(ent, 4), "nw": nw}
            pool.append(row); cf.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (k + 1) % 1000 == 0: print(f"  v5 {k+1}/{len(rows)} (kept {len(pool)})", flush=True)
    del M; torch.cuda.empty_cache()

# shortlist: top-SHORT by entropy per speaker
byspk = defaultdict(list)
for r in pool: byspk[r["video"]].append(r)
shortlist = []
for vid, cs in byspk.items():
    cs.sort(key=lambda x: -x["ent"]); shortlist.extend(cs[:SHORT])
print(f"[stage1] shortlist {len(shortlist)} clips over {len(byspk)} speakers", flush=True)

# ---------- Stage 2: Whisper-ft over shortlist ----------
print(f"[stage2] whisper {WCK}", flush=True)
proc = WhisperProcessor.from_pretrained("openai/whisper-medium", language="sanskrit", task="transcribe")
wm = WhisperForConditionalGeneration.from_pretrained(WCK).to("cuda").half().eval()
wm.generation_config.language = "sanskrit"; wm.generation_config.task = "transcribe"; wm.generation_config.forced_decoder_ids = None
asr = pipeline("automatic-speech-recognition", model=wm, tokenizer=proc.tokenizer,
               feature_extractor=proc.feature_extractor, device=0, torch_dtype=torch.float16)
B = 16
for i in range(0, len(shortlist), B):
    chunk = shortlist[i:i + B]
    wavs = [loadwav(r["audio_filepath"]) for r in chunk]
    outs = asr(wavs, batch_size=B, generate_kwargs=dict(language="sanskrit", task="transcribe"))
    for r, o in zip(chunk, outs): r["wh"] = o["text"]
    if (i + B) % 320 == 0: print(f"  whisper {min(i+B,len(shortlist))}/{len(shortlist)}", flush=True)

# ---------- Stage 3: disagreement -> hardest-first pick ----------
def looks_garbage(d):
    t = norm(d).replace(' ', '')
    return len(t) > 0 and len(set(t)) / len(t) < 0.30       # repeated-syllable noise
for r in shortlist:
    a = norm(r["draft"]).split(); b = norm(r.get("wh", "")).split()
    r["dis"] = ed(a, b) / max(1, max(len(a), len(b)))
# cache every scored shortlist clip so the cutoff can be re-tuned instantly (no Whisper re-run)
with open(f"{ADIR}/shortlist_scored.jsonl", "w", encoding="utf-8") as sf2:
    for r in shortlist:
        sf2.write(json.dumps({k: r.get(k) for k in ("audio_filepath", "video", "start", "dur",
                              "draft", "wh", "dis", "ent", "nw")}, ensure_ascii=False) + "\n")
ENGCAP = 0.90                                   # dis>this => ~total disagreement = English/noise, not hard Sanskrit
bys = defaultdict(list)
for r in shortlist:
    if r["dis"] > ENGCAP or looks_garbage(r["draft"]): continue   # exclude non-Sanskrit BEFORE ranking
    bys[r["video"]].append(r)
picked = []
for vid, cs in bys.items():
    cs.sort(key=lambda x: -x["dis"])            # HARDEST first
    acc = []
    for r in cs:                                # temporal spread
        if all(abs(r["start"] - a["start"]) >= MINGAP for a in acc): acc.append(r)
        if len(acc) >= PER_SPK: break
    for r in cs:                                # relax spread only if a speaker is short — still hardest available
        if len(acc) >= PER_SPK: break
        if r not in acc: acc.append(r)
    picked.extend(acc[:PER_SPK])

with open(OUTMAN, "w", encoding="utf-8") as f:
    for r in picked:
        bn = os.path.basename(r["audio_filepath"])
        shutil.copy(r["audio_filepath"], f"{CLIPS}/{bn}")
        f.write(json.dumps({"id": bn[:-4], "audio": bn, "video": r["video"], "dur": round(r["dur"], 2),
                            "draft": r["draft"], "wh": r.get("wh", ""), "dis": round(r["dis"], 3),
                            "ent": r["ent"]}, ensure_ascii=False) + "\n")
print(f"\nDONE: {len(picked)} clips over {len(bys)} speakers -> {OUTMAN}", flush=True)
ds = [r["dis"] for r in picked]
print(f"disagreement picked: median {np.median(ds):.2f}  min {min(ds):.2f}  max {max(ds):.2f}", flush=True)
