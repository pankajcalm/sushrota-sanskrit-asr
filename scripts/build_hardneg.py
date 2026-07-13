#!/usr/bin/env python3
"""Select hard-negative clips for scholar annotation: 20 per speaker (video-id), ranked by
CTC-posterior ENTROPY + OOV-rate (NOT confidence — proven not to separate right/wrong).
Runs v5 once over the full VAD-segment pool to get a draft + hardness signals, picks the
hardest 20/speaker with temporal spread, copies clips + writes a prefilled manifest."""
import os, json, re, shutil, unicodedata, pickle
from collections import defaultdict
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as nemo_asr

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
OFF, V, BLANK = 4096, 256, 5632
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
POOL = f"{ROOT}/data/epgp/manifest_train.jsonl"
OUTDIR = f"{ROOT}/data/epgp/annot_clips"; os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(f"{ROOT}/data/epgp/annot", exist_ok=True)
OUTMAN = f"{ROOT}/data/epgp/annot/manifest_hardneg.jsonl"
PER_SPK, MINGAP, DMIN, DMAX = 20, 10.0, 3.0, 12.0   # 20/spk, 10s temporal gap, 3–12s clips

KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def lse(a, ax): m = a.max(ax, keepdims=True); return m + np.log(np.exp(a - m).sum(ax, keepdims=True))

print(f"[boot] model on {DEV}", flush=True)
M = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location=DEV); M.eval()
LABELS = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
LEX = set(pickle.load(open(f"{ROOT}/data/lexicon.pkl", "rb")).keys())
print(f"[boot] lexicon {len(LEX)} forms", flush=True)
LOGN = np.log(V + 1)

def analyze(wav):
    sig = torch.tensor(wav).unsqueeze(0).to(DEV); sl = torch.tensor([len(wav)]).to(DEV)
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
        lp = M.ctc_decoder(encoder_output=enc)[0].detach().cpu().numpy()
    cols = [BLANK] + list(range(OFF, OFF + V)); P = lp[:, cols]; P = P - lse(P, 1)
    ids = P.argmax(1)
    probs = np.exp(P); ent = -(probs * P).sum(1) / LOGN         # per-frame entropy, normalized [0,1]
    emit = ids != 0
    mean_ent = float(ent[emit].mean()) if emit.any() else 0.0
    out = []; prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != 0: out.append(LABELS[i - 1])
        prev = i
    draft = ''.join(out).replace('▁', ' ').strip()
    words = norm(draft).split()
    oov = sum(1 for w in words if w not in LEX) / max(1, len(words))
    return draft, mean_ent, oov, len(words)

rows = [json.loads(l) for l in open(POOL)]
rows = [r for r in rows if DMIN <= r.get("dur", 0) <= DMAX]
print(f"[pool] {len(rows)} clips in [{DMIN},{DMAX}]s", flush=True)
cand = []
for k, r in enumerate(rows):
    try:
        wav, sr = sf.read(r["audio_filepath"], dtype="float32")
    except Exception:
        continue
    if wav.ndim > 1: wav = wav.mean(1)
    if len(wav) == 0 or float(np.sqrt(np.mean(wav ** 2))) < 0.006: continue   # skip silent/dead
    draft, ent, oov, nw = analyze(wav)
    if nw < 3: continue                                                       # need content to annotate
    cand.append({"audio_filepath": r["audio_filepath"], "video": r["video"],
                 "start": r.get("start", 0.0), "dur": r["dur"], "draft": draft,
                 "ent": ent, "oov": oov, "nw": nw})
    if (k + 1) % 1000 == 0: print(f".. scored {k+1}/{len(rows)} (kept {len(cand)})", flush=True)

# global z-scores -> hardness = z(entropy) + z(oov)
ents = np.array([c["ent"] for c in cand]); oovs = np.array([c["oov"] for c in cand])
ez = (ents - ents.mean()) / (ents.std() + 1e-9); oz = (oovs - oovs.mean()) / (oovs.std() + 1e-9)
for i, c in enumerate(cand): c["score"] = float(ez[i] + oz[i])

byspk = defaultdict(list)
for c in cand: byspk[c["video"]].append(c)
picked = []
for vid, cs in byspk.items():
    cs.sort(key=lambda x: -x["score"])
    acc = []
    for c in cs:                                     # hardest first, enforce temporal spread
        if all(abs(c["start"] - a["start"]) >= MINGAP for a in acc): acc.append(c)
        if len(acc) >= PER_SPK: break
    for c in cs:                                     # backfill if gap left us short
        if len(acc) >= PER_SPK: break
        if c not in acc: acc.append(c)
    picked.extend(acc[:PER_SPK])

with open(OUTMAN, "w", encoding="utf-8") as f:
    for c in picked:
        bn = os.path.basename(c["audio_filepath"])
        shutil.copy(c["audio_filepath"], f"{OUTDIR}/{bn}")
        f.write(json.dumps({"id": bn[:-4], "audio": bn, "video": c["video"], "dur": round(c["dur"], 2),
                            "draft": c["draft"], "ent": round(c["ent"], 3), "oov": round(c["oov"], 3),
                            "score": round(c["score"], 3)}, ensure_ascii=False) + "\n")
print(f"\nDONE: {len(picked)} clips over {len(byspk)} speakers -> {OUTMAN}", flush=True)
print(f"median score picked {np.median([c['score'] for c in picked]):.2f} vs pool {np.median([c['score'] for c in cand]):.2f}", flush=True)
