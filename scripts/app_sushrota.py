#!/usr/bin/env python3
"""Su-shrotaa backend — CPU FastAPI. /transcribe: audio -> transcript + per-word top-5
corpus suggestions (edit-distance-<=2 substitutions AND sandhi-splits, CTC-acoustic ranked).
/feedback: consented flywheel logging of (audio, raw, corrected). No GPU."""
import os; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import io, json, glob, re, unicodedata, pickle, time, uuid
from collections import Counter, defaultdict
import numpy as np, soundfile as sf, torch
from rapidfuzz import process, distance as rfdist
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import nemo.collections.asr as nemo_asr

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
OFF, V, BLANKCOL = 4096, 256, 0           # sa slice; column layout: 0=blank, 1..256 = token id 0..255
CONF_FLAG = 0.55
SPLIT_MARGIN = 8.0        # offer a split only if raw−split CTC gap is below this (model was torn = merge)
FLY = f"{ROOT}/data/flywheel"; os.makedirs(f"{FLY}/audio", exist_ok=True)
torch.set_num_threads(4)
KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
def norm(s):
    s = unicodedata.normalize('NFC', s); s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
LEVd = rfdist.Levenshtein.distance
def lse(a, axis):
    m = a.max(axis, keepdims=True); return (m + np.log(np.exp(a - m).sum(axis, keepdims=True)))

print("[boot] loading model...", flush=True)
M = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location="cpu")
M.eval()
LABELS = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))
def surf(tokid): return LABELS[tokid]

print("[boot] loading lexicon...", flush=True)
LEXPKL = f"{ROOT}/data/lexicon.pkl"
if os.path.exists(LEXPKL):
    LEX = pickle.load(open(LEXPKL, "rb"))
else:
    lex = Counter()
    for fn in glob.glob(f"{ROOT}/corpus/norm/*.txt"):
        if fn.endswith(('.prose.txt', '.verse.txt')): continue
        for line in open(fn, encoding='utf-8', errors='ignore'):
            for w in norm(line).split():
                if w: lex[w] += 1
    for gf in glob.glob(f"{ROOT}/corpus/glossary/rerank/*.hotwords.txt"):
        for line in open(gf, encoding='utf-8', errors='ignore'):
            if line.startswith('#') or not line.strip(): continue
            w = norm(line.split('\t')[0])
            if w: lex[w] += 50
    LEX = {w: c for w, c in lex.items() if c >= 2}
    pickle.dump(LEX, open(LEXPKL, "wb"))
BYLEN = defaultdict(list)
for w in LEX: BYLEN[len(w)].append(w)
print(f"[boot] lexicon {len(LEX)} forms. ready.", flush=True)

def sa_logprobs(wav):
    sig = torch.tensor(wav).unsqueeze(0); sl = torch.tensor([len(wav)])
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
        lp = M.ctc_decoder(encoder_output=enc)[0].detach().cpu().numpy()
    cols = [5632] + list(range(OFF, OFF + V))
    sub = lp[:, cols]
    return sub - lse(sub, 1)

_tk = {}
def toks(w):
    if w not in _tk: _tk[w] = M.tokenizer.text_to_ids(w, "sa")
    return _tk[w]

def ctc_score(seq, P):
    T, L = P.shape[0], len(seq)
    if L == 0 or T == 0: return -1e30
    ext = [BLANKCOL]
    for s in seq: ext += [s + 1, BLANKCOL]
    S = len(ext); NEG = -1e30
    a = np.full(S, NEG); a[0] = P[0, ext[0]]
    if S > 1: a[1] = P[0, ext[1]]
    for t in range(1, T):
        na = np.full(S, NEG)
        for s in range(S):
            v = a[s]
            if s > 0: v = np.logaddexp(v, a[s - 1])
            if s > 1 and ext[s] != BLANKCOL and ext[s] != ext[s - 2]: v = np.logaddexp(v, a[s - 2])
            na[s] = v + P[t, ext[s]]
        a = na
    return float(np.logaddexp(a[S - 1], a[S - 2])) if S > 1 else float(a[S - 1])

def edit2_subs(h):
    pool = []
    for L in range(len(h) - 2, len(h) + 3): pool.extend(BYLEN.get(L, ()))
    if not pool: return []
    return [w for w, d, _ in process.extract(h, pool, scorer=LEVd, score_cutoff=2, limit=80) if w != h]

def best_split(h):
    """Cleanest split of h into two real lexicon words -> 'w1 w2', else None.
    Used both to detect legitimate compounds (exempt from flagging) and, when a word
    IS flagged, as a candidate."""
    best = None
    if len(h) >= 6:
        for k in range(2, len(h) - 1):
            w1, w2 = h[:k], h[k:]
            if len(w1) >= 2 and len(w2) >= 2 and w1 in LEX and w2 in LEX:
                sc = LEX[w1] + LEX[w2]
                if best is None or sc > best[0]: best = (sc, w1 + " " + w2)
    return best[1] if best else None

def analyze(h, span, conf):
    """Precision-first flagging. Returns (suggestions, flagged).
    Flags only on genuine error evidence — never on bare OOV, never on legitimate compounds."""
    inlex = h in LEX
    lowconf = conf < CONF_FLAG
    if inlex and not lowconf:
        return [], False                       # known, confident -> trust
    if len(h) < 3:
        return [], False
    subs = edit2_subs(h)
    split = best_split(h)
    raw_score = ctc_score(toks(h), span)
    # a split is a real fix only if the audio supports a word boundary there (merge error),
    # not a genuine compound (audio prefers one word). Greedy omitted the break, so the split
    # scores below raw; a SMALL gap => the model was torn => merge error.
    split_ok = split is not None and (raw_score - ctc_score(toks(split), span)) < SPLIT_MARGIN
    is_compound = split is not None and not split_ok      # decomposes into real words, audio says one word
    close = len(subs) > 0                                 # a real word within edit-2 -> typo-like
    typo = (not inlex) and (not is_compound) and close
    if not (lowconf or typo or split_ok):
        return [], False                                  # well-formed / rare-but-correct word -> leave alone
    cands = set(subs)
    if split_ok: cands.add(split)
    ranked = sorted(cands, key=lambda c: -ctc_score(toks(c), span))[:5]
    return ranked, len(ranked) > 0

def transcribe_words(wav, interim=False):
    P = sa_logprobs(wav)
    ids = P.argmax(1)
    emitted, prev = [], -1
    for t, i in enumerate(ids):
        i = int(i)
        if i != prev and i != BLANKCOL: emitted.append((t, i - 1, float(P[t, i])))
        prev = i
    words, cur = [], []
    def flush(end):
        if cur:
            text = ''.join(surf(k) for _, k, _ in cur).replace('▁', ' ').strip()
            conf = float(np.exp(np.mean([lp for _, _, lp in cur])))
            words.append({"text": text, "start": cur[0][0], "end": end, "conf": round(conf, 3)})
    for (t, k, lp) in emitted:
        if surf(k).startswith('▁') and cur: flush(t); cur.clear()
        cur.append((t, k, lp))
    flush(len(ids))
    for w in words:
        if interim:                            # streaming tick: raw text only, skip suggestions (fast)
            w["flagged"] = False; w["suggestions"] = []
        else:
            sugg, flag = analyze(w["text"], P[w["start"]:max(w["start"] + 1, w["end"])], w["conf"])
            w["suggestions"] = sugg
            w["flagged"] = flag
        del w["start"]; del w["end"]
    return words

app = FastAPI(title="Su-shrotaa ASR")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return {"status": "ok", "lexicon": len(LEX)}

def is_degenerate(words):
    """Detect a hallucinated/looping decode: low lexical variety AND a 3-word phrase that
    repeats >=3x within the segment. Conservative on purpose — genuine japa/refrains repeat
    short mantras, but a full trigram looping 3x in one <=15s segment is the runaway signature."""
    toks = [w["text"] for w in words if w.get("text")]
    n = len(toks)
    if n < 8: return False
    if len(set(toks)) / n >= 0.5: return False               # enough variety -> real speech
    tri = Counter(tuple(toks[i:i+3]) for i in range(n - 2))
    return bool(tri) and tri.most_common(1)[0][1] >= 3

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), interim: str = Form("false")):
    raw = await audio.read()
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != 16000:
        import librosa; wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    if wav.size == 0 or float(np.sqrt(np.mean(wav ** 2))) < 0.004:   # dead/silent capture -> no hallucination
        return {"raw_text": "", "words": [], "note": "no speech detected"}
    isint = (interim == "true")
    words = transcribe_words(wav, interim=isint)
    if (not isint) and is_degenerate(words):                        # suppress runaway loop on final decode
        return {"raw_text": "", "words": [], "note": "unclear — please try again"}
    return {"raw_text": " ".join(w["text"] for w in words), "words": words}

@app.post("/feedback")
async def feedback(kind: str = Form("final"), session_id: str = Form(""),
                   seg_id: str = Form(""), raw_text: str = Form(""),
                   corrected_text: str = Form(""), suggest: str = Form(""),
                   consent: str = Form("false"), audio: UploadFile = File(None)):
    if consent != "true":
        return {"stored": False, "reason": "no consent"}
    sess = session_id or uuid.uuid4().hex[:12]
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec = {"kind": kind, "session": sess, "ts": ts, "suggest": suggest}
    if kind == "segment":
        rec["seg"] = seg_id; rec["raw"] = raw_text
        if audio is not None:
            ap = f"{FLY}/audio/{sess}_{seg_id or uuid.uuid4().hex[:6]}.wav"
            with open(ap, "wb") as f: f.write(await audio.read())
            rec["audio"] = ap
    else:
        rec["corrected"] = corrected_text
    with open(f"{FLY}/events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"stored": True}

# --- gold transcription tool (internal) ---
GOLDDIR = f"{ROOT}/data/epgp"
if os.path.isdir(f"{GOLDDIR}/eval_clips"):
    app.mount("/gold_clips", StaticFiles(directory=f"{GOLDDIR}/eval_clips"), name="gold_clips")

@app.get("/gold", response_class=HTMLResponse)
def gold_page():
    return open(f"{ROOT}/scripts/goldtool.html", encoding="utf-8").read()

@app.get("/gold/manifest")
def gold_manifest():
    saved = {}
    sp = f"{GOLDDIR}/gold_refs.jsonl"
    if os.path.exists(sp):
        for l in open(sp, encoding="utf-8"):
            try: r = json.loads(l); saved[r["id"]] = r["text"]
            except: pass
    rows = []
    for l in open(f"{GOLDDIR}/manifest_gold.jsonl", encoding="utf-8"):
        g = json.loads(l)
        rows.append({"id": g["id"], "audio": "/gold_clips/" + os.path.basename(g["audio_filepath"]),
                     "prefill": g.get("prefill", ""), "dur": g.get("dur"),
                     "done": g["id"] in saved, "saved": saved.get(g["id"])})
    return {"clips": rows, "saved": len(saved), "total": len(rows)}

@app.post("/gold/save")
async def gold_save(id: str = Form(...), text: str = Form(...)):
    with open(f"{GOLDDIR}/gold_refs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": id, "text": text, "ts": time.strftime('%Y-%m-%dT%H:%M:%S')}, ensure_ascii=False) + "\n")
    return {"ok": True}

@app.get("/stats")
def stats():
    import os as _os
    n_seg = n_fin = 0; sessions = set()
    p = f"{FLY}/events.jsonl"
    if _os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            try: r = json.loads(line)
            except: continue
            sessions.add(r.get("session"))
            if r.get("kind") == "segment": n_seg += 1
            else: n_fin += 1
    return {"sessions": len(sessions), "segments": n_seg, "finals": n_fin}

# --- multi-scholar annotation portal (hard-negative clips) ---
ANNOTDIR = f"{ROOT}/data/epgp/annot"
if os.path.isdir(f"{ROOT}/data/epgp/annot_clips"):
    app.mount("/annot_clips", StaticFiles(directory=f"{ROOT}/data/epgp/annot_clips"), name="annot_clips")
import threading
_ANNOT_MAN = None
_CLAIMS = {}                                  # id -> (scholar, ts); a soft lock so scholars don't collide
_ANNOT_LOCK = threading.Lock()                # serialize hand-out so 3–4 concurrent annotators never get the same clip
_CLAIM_TTL = 900                              # a claimed-but-unsaved clip is re-offered only after 15 min idle
def _annot_manifest():
    global _ANNOT_MAN
    if _ANNOT_MAN is None:
        p = f"{ANNOTDIR}/manifest_hardneg.jsonl"
        _ANNOT_MAN = [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []
    return _ANNOT_MAN
def _annot_done():
    done = set(); p = f"{ANNOTDIR}/annot_refs.jsonl"
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try: done.add(json.loads(l)["id"])
            except: pass
    return done

@app.get("/annot", response_class=HTMLResponse)
def annot_page():
    return open(f"{ROOT}/scripts/annot.html", encoding="utf-8").read()

@app.get("/annot/next")
def annot_next(scholar: str = ""):
    with _ANNOT_LOCK:                                                                  # atomic hand-out
        man = _annot_manifest(); done = _annot_done(); now = time.time()
        for k in [k for k, v in _CLAIMS.items() if now - v[1] > _CLAIM_TTL]: _CLAIMS.pop(k, None)
        for c in man:
            cid = c["id"]
            if cid in done: continue                                                   # saved -> never shown again
            cl = _CLAIMS.get(cid)
            if cl and cl[0] != scholar and now - cl[1] < _CLAIM_TTL: continue           # in progress by someone else
            _CLAIMS[cid] = (scholar, now)
            return {"id": cid, "audio": "/annot_clips/" + c["audio"], "draft": c.get("draft", ""),
                    "video": c.get("video"), "dur": c.get("dur"), "done": len(done), "total": len(man)}
        return {"id": None, "done": len(done), "total": len(man)}

@app.post("/annot/save")
async def annot_save(id: str = Form(...), scholar: str = Form(""), text: str = Form(""), unclear: str = Form("false")):
    with open(f"{ANNOTDIR}/annot_refs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": id, "scholar": scholar, "text": text, "unclear": unclear == "true",
                            "ts": time.strftime('%Y-%m-%dT%H:%M:%S')}, ensure_ascii=False) + "\n")
    _CLAIMS.pop(id, None)
    return {"ok": True}

@app.get("/annot/progress")
def annot_progress():
    man = _annot_manifest(); done = set(); by = {}
    p = f"{ANNOTDIR}/annot_refs.jsonl"
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try: r = json.loads(l)
            except: continue
            done.add(r["id"]); sc = r.get("scholar", "?"); by[sc] = by.get(sc, 0) + 1
    return {"done": len(done), "total": len(man), "by_scholar": by}
