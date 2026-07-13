#!/usr/bin/env python3
"""Su-shrotā Abhyāsa — standalone chanting-practice backend.

User pastes a text (any script) -> we transliterate to Devanāgarī, segment into aksharas
grouped by line/half-line. User selects any span (tap start/end akshara) and chants it.
We force-align the selected reference to their audio using v5 (same model, sa-slice
posteriors) and score each akshara by GOP (NORM = target_logpost - top_logpost, aggregated
MIN over sub-tokens). Green/amber/red bands + a % correctness. Validated in gop_phase15
(hard-confusable AUC 0.967).

Endpoints:  GET /  ,  POST /prep {text}  ,  POST /score {audio, text}  ,  GET /health
Run:  CUDA_VISIBLE_DEVICES=0 python app_practice.py   (uvicorn on :8010)
"""
import os, io, re, json, time, uuid, hashlib, unicodedata
import numpy as np, soundfile as sf, torch, requests
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import nemo.collections.asr as nemo_asr
from indic_transliteration import sanscript, detect
from indic_transliteration.sanscript import transliterate

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
OFF, V, BL = 4096, 256, 5632
NEG = -1e30
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
# GOP bands (gop_phase15 recalibrated w/ blank-excluded top): correct syllables hug 0,
# mispron median -6.9. RED<-2.5 catches ~90% errors flagging <2% correct. correctness ramps linearly.
GREEN_THR, RED_THR = -0.5, -2.5
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8020")   # Vagdhenu microservice (tts_api.py)
CACHE = f"{ROOT}/data/tts_cache"; os.makedirs(CACHE, exist_ok=True)
FB = f"{ROOT}/data/practice_feedback"; os.makedirs(f"{FB}/audio", exist_ok=True)   # "I said it right" logs
FLYW = f"{ROOT}/data/practice_flywheel"; os.makedirs(f"{FLYW}/audio", exist_ok=True)  # consented ASR data
COLLECT_MIN = float(os.environ.get("COLLECT_MIN", "90"))   # collect (audio, reference) pairs at >= this %
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "10"))     # new ślokas synthesised per IP per day
LIMITS_FILE = f"{ROOT}/data/practice_limits.json"
try: _limits = json.load(open(LIMITS_FILE))
except Exception: _limits = {}
def client_ip(request):
    h = request.headers
    return (h.get("cf-connecting-ip") or h.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "?") or "?")
def limit_check(ip):
    """Return (allowed, remaining) and increment the IP's daily counter. Call only when real
    synthesis will happen (cached re-generations are free)."""
    key = f"{ip}|{time.strftime('%Y-%m-%d')}"
    n = _limits.get(key, 0)
    if n >= DAILY_LIMIT: return False, 0
    _limits[key] = n + 1
    try: json.dump(_limits, open(LIMITS_FILE, "w"))
    except Exception: pass
    return True, DAILY_LIMIT - (n + 1)

KEEP = re.compile(r'[^ऀ-ॿ\s]'); DROP = re.compile(r'[०-९।॥ऽॐ॒॑॓॔᳐-᳿]')
NASAL = re.compile(r'म्(?=\s|[।॥]|$)')            # word-final म् (coda /m/) == anusvāra
def dedup_marks(s):
    """Collapse consecutive identical combining marks — a CTC greedy artifact where a held
    sound emits [mātrā, blank, mātrā] and survives collapse as a doubled sign (सुु -> सु).
    Never touches repeated base letters (न न); no valid Devanāgarī stacks a vowel sign twice."""
    out = []
    for ch in s:
        if out and ch == out[-1] and unicodedata.category(ch) in ('Mn', 'Mc'):
            continue
        out.append(ch)
    return ''.join(out)
def canon(s):
    """Output/target canonicalisation shared by reference, ASR 'heard', and scoring: dedup the
    CTC doubled marks, and treat word-final म् (coda /m/) as anusvāra — देवम् == देवं, किम् == किं.
    (Restricted to word-final: a mid-word anusvāra before a non-labial is a different nasal.)"""
    return NASAL.sub('ं', dedup_marks(s))
def norm(s):
    s = unicodedata.normalize('NFC', s); s = canon(s)
    s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
def lse(x, ax):
    mx = x.max(ax, keepdims=True); return mx + np.log(np.exp(x - mx).sum(ax, keepdims=True))
def is_base(ch):
    o = ord(ch)
    return (0x0905 <= o <= 0x0939) or (0x0958 <= o <= 0x0961) or (0x0972 <= o <= 0x097F)

print("[boot] loading v5 on", DEV, flush=True)
M = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
    f"{ROOT}/exp/ft_ctc_v5/ft_ctc_ep20.nemo", map_location=DEV).eval()
SUB = M.tokenizer.tokenizers_dict["sa"]
LAB = json.load(open(f"{ROOT}/data/eval_logits/labels.json"))   # col i (1..256) -> LAB[i-1]
# Representational-invariance for GOP: the target token's effective posterior is the MAX over
# an equivalence set, so the model's arbitrary spelling of the SAME sound never costs the learner.
#  (a) word-boundary: word-initial ▁प == mid-word प.
#  (b) nasal coda: anusvāra ं / candrabindu ँ / halant-nasals म् न् ण् ञ् ङ् are all "a nasal";
#      any of them satisfies a nasal-bearing target (the model writes word-final /m/ as म्, not ं).
_surf2col = {LAB[d]: d + 1 for d in range(min(V, len(LAB)))}
_CODA = {'ं', 'ँ', 'म्', 'न्', 'ण्', 'ञ्', 'ङ्'}          # pure nasal-coda tokens only
_CODACOLS = [c for s, c in _surf2col.items() if s in _CODA]
def _is_nasal(s): return s in _CODA                        # equivalence ONLY for a standalone coda
EQUIV = [np.array([c], dtype=int) for c in range(V + 1)]
for d in range(min(V, len(LAB))):
    c = d + 1; s = LAB[d]; eq = {c}
    sib = s[1:] if s.startswith('▁') else '▁' + s
    if sib in _surf2col: eq.add(_surf2col[sib])
    if _is_nasal(s): eq.update(_CODACOLS)
    EQUIV[c] = np.array(sorted(eq), dtype=int)
print("[boot] ready", flush=True)

def greedy(P, blank_pen=0.0):
    """Free (unconstrained) CTC decode -> what v5 heard. blank_pen subtracts from the blank column
    before argmax (higher => fewer deletions); used to probe decode stability for confidence."""
    Q = P
    if blank_pen: Q = P.copy(); Q[:, 0] = Q[:, 0] - blank_pen
    ids = Q.argmax(1); o = []; prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != 0: o.append(LAB[i - 1])
        prev = i
    return canon(''.join(o).replace('▁', ' ')).strip()

# ---- script registry: any-script <-> Devanāgarī (Devanāgarī is the internal canonical) ----
def _sc(n): return getattr(sanscript, n, None)
SCHEME = {"devanagari": None, "iast": _sc("IAST"), "itrans": _sc("ITRANS"), "hk": _sc("HK"),
          "slp1": _sc("SLP1"), "kolkata": _sc("IAST"), "velthuis": _sc("VELTHUIS"),
          "kannada": _sc("KANNADA"), "telugu": _sc("TELUGU"), "tamil": _sc("TAMIL"),
          "malayalam": _sc("MALAYALAM"), "bengali": _sc("BENGALI"), "gujarati": _sc("GUJARATI"),
          "gurmukhi": _sc("GURMUKHI"), "oriya": _sc("ORIYA"), "grantha": _sc("GRANTHA"),
          "sinhala": _sc("SINHALA")}
SCHEME = {k: v for k, v in SCHEME.items() if k == "devanagari" or v is not None}
NAMES = {"devanagari": "Devanāgarī", "kannada": "Kannada ಕನ್ನಡ", "telugu": "Telugu తెలుగు",
         "tamil": "Tamil தமிழ்", "malayalam": "Malayalam മലയാളം", "bengali": "Bengali বাংলা",
         "gujarati": "Gujarati ગુજરાતી", "gurmukhi": "Gurmukhi ਗੁਰਮੁਖੀ", "oriya": "Odia ଓଡ଼ିଆ",
         "grantha": "Grantha", "sinhala": "Sinhala සිංහල", "iast": "IAST (Roman)",
         "itrans": "ITRANS (Roman)", "hk": "Harvard-Kyoto (Roman)", "slp1": "SLP1 (Roman)",
         "kolkata": "IAST/Kolkata", "velthuis": "Velthuis (Roman)"}
ROMAN_IDS = {"iast", "itrans", "hk", "slp1", "kolkata", "velthuis"}
# scripts offered in the override picker (skip near-duplicate roman variants)
PICKER = [i for i in SCHEME if i not in ("kolkata", "velthuis")]

def detect_script_id(text):
    """Best-guess script id. Latin letters => a Roman scheme; else an Indic block; fallbacks safe."""
    if re.search(r'[A-Za-z]', text):
        try: s = str(detect.detect(re.sub(r'[।॥\s]+', ' ', text))).lower()
        except Exception: s = "iast"
        return s if (s in SCHEME and s in ROMAN_IDS) else "iast"
    try: s = str(detect.detect(text)).lower()
    except Exception: s = "devanagari"
    return s if s in SCHEME else "devanagari"

def to_devanagari(text, script=None):
    """Input (in `script`, or auto-detected) -> Devanāgarī (internal canonical)."""
    sid = script if script in SCHEME else detect_script_id(text)
    sch = SCHEME.get(sid)
    if sch is None: return text                                # already Devanāgarī
    try: return transliterate(text, sch, sanscript.DEVANAGARI)
    except Exception: return text

def from_dev(dev, script):
    """Devanāgarī -> display script (identity for Devanāgarī / unknown)."""
    sch = SCHEME.get(script)
    if sch is None: return dev
    try: return transliterate(dev, sanscript.DEVANAGARI, sch)
    except Exception: return dev

# ---- vṛtta (metre) detection for pāda splitting (inlined from vagdhenu chandas_labeler) ----
SHORT = set("aiufx"); LONG = set("AIUFXeEoO"); VOW = SHORT | LONG; MK = set("MH~")
SIGNATURES = {
    "GGLGGLLGLGG": "indravajrā", "LGLGGLLGLGG": "upendravajrā", "LGLGGLLGLGLG": "vaṃśastha",
    "GGLGLLLGLLGLGG": "vasantatilakā", "LLLLLLGGGLGGLGG": "mālinī",
    "GGGGLLLLLGGLGGLGG": "mandākrāntā", "LGGGGGLLLLLGGLLLG": "śikhariṇī",
    "LGLLLGLGLLLGLGGLG": "pṛthvī", "GGGLLGLGLLLGGGLGGLG": "śārdūlavikrīḍita",
    "GGGGLGGLLLLLLGGLGGLGG": "sragdharā",
}
def scan_weights(dev_text):
    """Per-akṣara laghu(L)/guru(G) scansion of Devanāgarī via SLP1 (chandas rules)."""
    s = "".join(c for c in transliterate(dev_text, sanscript.DEVANAGARI, sanscript.SLP1)
                if c.isalpha() or c in MK)
    vp = [i for i, c in enumerate(s) if c in VOW]; w = []
    for k, i in enumerate(vp):
        nxt = vp[k + 1] if k + 1 < len(vp) else len(s); between = s[i + 1:nxt]
        if s[i] in LONG or any(m in between for m in MK): w.append("G")
        else:
            w.append("G" if sum(c not in VOW and c not in MK for c in between) >= 2 else "L")
    return w
def _match(ww, sig):                                   # exact but pādānta anceps (last free)
    return len(ww) == len(sig) and all(a == b for a, b in zip(ww[:-1], sig[:-1]))
def _quarters_equal(w, parts, L):
    q = [w[p * L:(p + 1) * L] for p in range(parts)]
    return all(_match(q[0], "".join(x)) or _match(x, "".join(q[0])) for x in q[1:])
def detect_pada_len(w):
    """Return (pada_len, metre_label) or None. Identify a sama-vṛtta by matching the FIRST pāda to a
    known L/G signature (tolerant of minor metrical liberties in later pādas — real verses take
    them); else anuṣṭubh by syllable count; else generic identical-quarters; else prose."""
    N = len(w)
    for sig, name in SIGNATURES.items():                # match pāda 1 to a known metre
        L = len(sig)
        if N % L == 0 and N // L in (1, 2, 4) and _match(w[:L], sig):
            return L, name
    if N in (8, 16, 32): return 8, "anuṣṭubh"           # dominant metre; no fixed L/G signature
    for parts in (4, 2):                                # generic sama-vṛtta: identical quarters
        if N % parts == 0:
            L = N // parts
            if 5 <= L <= 30 and _quarters_equal(w, parts, L): return L, f"sama-{L}"
    return None

def _cluster(text):
    """Single source of truth for akṣara segmentation. Returns [{text, word_end, toks}]:
    grapheme-cluster aksharas over the sa-tokenizer surface (daṇḍa stripped, spaces = word
    boundaries), vowel-less codas merged into the previous akṣara, and the sa-token indices
    composing each akṣara. Used by BOTH /prep (display) and /score (GOP), so they stay 1:1."""
    t = norm(text)
    if not t: return []
    chars = []
    for j, s in enumerate(SUB.text_to_tokens(t)):
        for ch in s: chars.append((' ', -1) if ch == '▁' else (ch, j))
    out = []; cur = ""; toks = []; prev_vir = False
    def flush(we):
        nonlocal cur, toks
        if cur: out.append({"text": cur, "word_end": we, "toks": sorted(set(toks))})
        cur = ""; toks = []
    for ch, tj in chars:
        if ch == ' ': flush(True); prev_vir = False; continue
        if is_base(ch) and cur and not prev_vir:
            flush(False); cur = ch; toks = [tj] if tj >= 0 else []
        else:
            cur += ch
            if tj >= 0: toks.append(tj)
        prev_vir = (ord(ch) == 0x094D)
    flush(True)
    merged = []                                         # merge vowel-less codas (…्) into prev
    for a in out:
        if merged and a["text"].endswith("्"):
            merged[-1]["text"] += a["text"]; merged[-1]["word_end"] = a["word_end"]
            merged[-1]["toks"] = sorted(set(merged[-1]["toks"]) | set(a["toks"]))
        else:
            merged.append(a)
    return merged

def akshara_split(text):
    return [{"text": a["text"], "word_end": a["word_end"]} for a in _cluster(text)]

def segment(dev_text):
    """Split into pādas. Verses are split on ॥ / blank line; within a verse, metre detection
    (detect_pada_len) sets the pāda boundaries. Prose / undetected verses fall back to
    daṇḍa/newline chunks. Returns list of pādas: {i, verse, metre, aksharas:[...]}."""
    verses = re.split(r'॥|\n\s*\n', dev_text)
    padas = []; vi = 0
    for verse in verses:
        if not norm(verse): continue
        aks = akshara_split(verse)
        if not aks: continue
        w = scan_weights(verse)
        det = detect_pada_len(w) if len(w) == len(aks) else None
        if det:
            L, metre = det
            for p0 in range(0, len(aks), L):
                chunk = aks[p0:p0 + L]
                if chunk: padas.append({"i": len(padas), "verse": vi, "metre": metre, "aksharas": chunk})
        else:                                          # prose / unknown -> daṇḍa & newline chunks
            for part in re.split(r'[।\n]+', verse):
                ca = akshara_split(part)
                if ca: padas.append({"i": len(padas), "verse": vi, "metre": "gadya", "aksharas": ca})
        vi += 1
    for p in padas:                                    # word-end on last akṣara of each pāda
        if p["aksharas"]: p["aksharas"][-1]["word_end"] = True
    return padas

def _join(plist, sep):
    return sep.join("".join(a["text"] + (" " if a["word_end"] else "") for a in p["aksharas"]).strip()
                    for p in plist)
def build_units(padas, granularity):
    """Group pādas into learning units. pada=each pāda; ardha=consecutive pairs; full=whole verse
    (per verse). Each unit: flat aksharas (display/score), tts_text (pādas joined by \\n for TTS
    prosody), text (score, pādas joined by space), metre (TTS hint)."""
    byverse = {}
    for p in padas: byverse.setdefault(p["verse"], []).append(p)
    groups = []                                        # (metre, verse, [pāda,...])
    for vi, pl in byverse.items():
        m = pl[0]["metre"]
        if granularity == "pada":    groups += [(m, vi, [p]) for p in pl]
        elif granularity == "ardha": groups += [(m, vi, pl[k:k + 2]) for k in range(0, len(pl), 2)]
        else:                        groups.append((m, vi, pl))
    units = []
    for i, (m, vi, pl) in enumerate(groups):
        units.append({"i": i, "verse": vi, "metre": m,
                      "aksharas": [a for p in pl for a in p["aksharas"]],
                      "tts_text": _join(pl, "\n"), "text": _join(pl, " ")})
    return units

def _tts_key(text, meter, seed=60): return hashlib.sha1(f"{text}|{meter}|{seed}".encode("utf-8")).hexdigest()[:16]
def is_cached(text, meter, seed=60): return os.path.exists(f"{CACHE}/{_tts_key(text, meter, seed)}.wav")
def tts_fetch(text, meter, seed=60):
    """Return a cached wav URL for (text, meter, seed), rendering via the Vagdhenu microservice
    once. Returns None if TTS is unavailable (UI degrades: chant still works, no reference audio)."""
    key = _tts_key(text, meter, seed)
    path = f"{CACHE}/{key}.wav"
    if not os.path.exists(path):
        try:
            r = requests.post(f"{TTS_URL}/tts", data={"text": text, "meter": meter, "seed": seed}, timeout=120)
            if r.status_code != 200: return None
            with open(path, "wb") as f: f.write(r.content)
        except Exception:
            return None
    return f"/tts_cache/{key}.wav"

def posteriors(wav):
    sig = torch.tensor(wav).unsqueeze(0).to(DEV); sl = torch.tensor([len(wav)]).to(DEV)
    with torch.no_grad():
        enc, _ = M.forward(input_signal=sig, input_signal_length=sl)
        lp = M.ctc_decoder(encoder_output=enc)[0].cpu().numpy()
    cols = [BL] + list(range(OFF, OFF + V)); P = lp[:, cols]
    return P - lse(P, 1)

def align_tokens(P, ids):
    """Force-align; return per-token (norm_gop, nframes) or None."""
    T = P.shape[0]
    cols = [d + 1 for d in ids if 0 <= d < V]; L = len(cols)
    if L == 0 or T < L: return None
    ext = [0]
    for c in cols: ext += [c, 0]
    ext = np.array(ext); S = len(ext); topcol = P[:, 1:].max(1)   # top over REAL tokens (exclude
    frame_lp = P[:, ext].astype(float, copy=True)                 # blank col 0: blank=continuation,
    for k in range(S):                                            # not competing evidence vs target;
        c = ext[k]                                                # target = max over its equivalence set
        if c != 0: frame_lp[:, k] = P[:, EQUIV[c]].max(1)
    allow = np.zeros(S, bool); allow[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), np.int32)
    dp[0, 0] = frame_lp[0, 0]
    if S > 1: dp[0, 1] = frame_lp[0, 1]
    for t in range(1, T):
        stay = dp[t - 1]
        prev = np.full(S, NEG); prev[1:] = dp[t - 1, :-1]
        skip = np.full(S, NEG); skip[2:] = dp[t - 1, :-2]; skip[~allow] = NEG
        cand = np.stack([stay, prev, skip]); arg = cand.argmax(0)
        dp[t] = cand.max(0) + frame_lp[t]; bp[t] = np.arange(S) - arg
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    if dp[T - 1, s] <= NEG / 2: return None
    path = np.empty(T, np.int32)
    for t in range(T - 1, -1, -1): path[t] = s; s = bp[t, s]
    out = []
    for j in range(L):
        fr = np.where(path == 2 * j + 1)[0]
        out.append((float((frame_lp[fr, 2 * j + 1] - topcol[fr]).mean()), len(fr)) if len(fr) else (NEG, 0))
    return out

def align_aksharas(ref, hyp):
    """Edit-distance align the ASR hyp aksharas to the reference aksharas (by canonical text).
    Returns per-reference-akshara (op, heard): op ∈ match/sub/del; heard = the hyp akshara that
    replaced it (for a substitution). Both sides are already canonicalised by _cluster→norm
    (dedup + म्≡ं + spacing), so equivalent spellings match."""
    R = [a["text"] for a in ref]; H = [a["text"] for a in hyp]
    n, m = len(R), len(H)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        Ri = R[i - 1]
        for j in range(1, m + 1):
            c = 0 if Ri == H[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + c, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j = n, m; ops = [("del", "")] * n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if R[i - 1] == H[j - 1] else 1):
            ops[i - 1] = ("match", "") if R[i - 1] == H[j - 1] else ("sub", H[j - 1]); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops[i - 1] = ("del", ""); i -= 1
        else:
            j -= 1
    return ops

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
from fastapi.staticfiles import StaticFiles
app.mount("/tts_cache", StaticFiles(directory=CACHE), name="tts_cache")
_SPIKE = f"{ROOT}/data/tts_spike"
if os.path.isdir(_SPIKE):
    app.mount("/spike_wav", StaticFiles(directory=_SPIKE), name="spike_wav")

@app.get("/health")
def health():
    try: tts = requests.get(f"{TTS_URL}/health", timeout=3).status_code == 200
    except Exception: tts = False
    return {"status": "ok", "device": DEV, "tts": tts}

@app.post("/detect")
async def detect_script(text: str = Form(...), script: str = Form("")):
    """Detect the input script (or honour an override), and echo the text back in that script
    (round-trip via Devanāgarī) so the user can confirm we read it correctly."""
    if not text.strip(): return JSONResponse({"error": "empty text"}, status_code=400)
    sid = script if script in SCHEME else detect_script_id(text)
    dev = to_devanagari(text, sid)
    return {"script": sid, "name": NAMES.get(sid, sid), "echo": from_dev(dev, sid),
            "devanagari": dev, "scripts": [{"id": i, "name": NAMES.get(i, i)} for i in PICKER]}

@app.post("/generate")
async def generate(request: Request, text: str = Form(...), script: str = Form("")):
    """Segment into pādas (metre-detected), then render ALL three levels — pāda/ardha/full, each
    a SEPARATE Vagdhenu pass over its own text (never audio-joined). Streams NDJSON progress:
    {plan,total} → {progress,done,total} per render → {done,levels}. Daily per-IP limit counts
    only NEW synthesis (cached re-generations are free)."""
    sid = script if script in SCHEME else detect_script_id(text)
    dev = to_devanagari(text, sid)
    padas = segment(dev)
    if not padas: return JSONResponse({"error": "no text found"}, status_code=400)
    levels = {}
    for gran in ("pada", "ardha", "full"):
        units = build_units(padas, gran)
        for u in units:
            for a in u["aksharas"]: a["disp"] = from_dev(a["text"], sid)
        levels[gran] = units
    jobs = {}                                                     # unique (tts_text -> metre)
    for lv in levels.values():
        for u in lv: jobs.setdefault(u["tts_text"], u["metre"])
    new = [t for t, m in jobs.items() if not is_cached(t, m)]
    if new:                                                       # charge quota only for real synthesis
        ok, remaining = limit_check(client_ip(request))
        if not ok:
            return JSONResponse({"error": "limit", "message":
                f"You've reached today's limit of {DAILY_LIMIT} ślokas from this network. Please come back tomorrow 🙏"},
                status_code=429)
    else:
        remaining = None

    def stream():
        yield json.dumps({"type": "plan", "total": len(jobs), "script": sid,
                          "name": NAMES.get(sid, sid), "remaining": remaining}, ensure_ascii=False) + "\n"
        wav = {}; done = 0
        for t, m in jobs.items():
            wav[t] = tts_fetch(t, m); done += 1
            yield json.dumps({"type": "progress", "done": done, "total": len(jobs)}) + "\n"
        for lv in levels.values():
            for u in lv: u["wav_url"] = wav.get(u["tts_text"]); u.pop("tts_text", None)
        tts_ok = any(u["wav_url"] for lv in levels.values() for u in lv)
        yield json.dumps({"type": "done", "devanagari": dev, "script": sid, "name": NAMES.get(sid, sid),
                          "levels": levels, "tts_ok": tts_ok}, ensure_ascii=False) + "\n"
    return StreamingResponse(stream(), media_type="application/x-ndjson")

@app.get("/spike", response_class=HTMLResponse)
def spike_page():
    rows = [("Full śloka", "वसुदेवसुतं देवं कंसचाणूरमर्दनम् । देवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥", "full"),
            ("Ardha 1", "वसुदेवसुतं देवं कंसचाणूरमर्दनम्", "ardha1"),
            ("Ardha 2", "देवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम्", "ardha2"),
            ("Pāda 1", "वसुदेवसुतं देवं", "pada1"), ("Pāda 2", "कंसचाणूरमर्दनम्", "pada2"),
            ("Pāda 3", "देवकीपरमानन्दं", "pada3"), ("Pāda 4", "कृष्णं वन्दे जगद्गुरुम्", "pada4")]
    body = "".join(
        f'<div style="margin:14px 0"><div style="font-size:13px;color:#888">{lab}</div>'
        f'<div style="font-family:serif;font-size:24px;margin:2px 0">{txt}</div>'
        f'<audio controls src="/spike_wav/{fn}.wav" style="width:100%"></audio></div>'
        for lab, txt, fn in rows)
    return f'<div style="max-width:640px;margin:24px auto;font-family:sans-serif;padding:0 16px">' \
           f'<h2>Vagdhenu render spike — pada vs ardha vs full</h2>{body}</div>'

@app.post("/prep")
async def prep(text: str = Form(...)):
    """Segment into pādas via metre detection. Each pāda carries its aksharas so the frontend
    selects whole pādas (min unit = 1 pāda) but colours per-syllable feedback."""
    dev = to_devanagari(text)
    padas = segment(dev)
    if not padas: return JSONResponse({"error": "no Devanāgarī text found"}, status_code=400)
    for p in padas:
        p["text"] = "".join(x["text"] + (" " if x["word_end"] else "") for x in p["aksharas"]).strip()
    return {"devanagari": dev, "padas": padas}

@app.post("/score")
async def score(audio: UploadFile = File(...), text: str = Form(...), script: str = Form(""),
                mode: str = Form("strict")):
    """Akshara-level text comparison: free-decode the chant, canonicalise both sides, align to the
    reference. match→green, substitution→red (with what was heard), deletion→amber (not caught).
    text = the unit's Devanāgarī (spaces at word ends); `heard` is echoed back in `script`."""
    raw = await audio.read()
    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as e:
        return JSONResponse({"error": f"could not decode audio: {e}"}, status_code=400)
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != 16000:
        idx = (np.arange(int(len(wav) * 16000 / sr)) * sr / 16000).astype(int)
        wav = wav[np.clip(idx, 0, len(wav) - 1)]
    # pre/post-roll silence: v5 tends to clip the audio onset/tail -> pad so the ASR doesn't drop
    # the first/last aksharas (which would show as spurious deletions).
    pad = np.zeros(int(0.3 * 16000), np.float32)
    wav = np.concatenate([pad, wav.astype(np.float32), pad])
    t = norm(text)
    if not t or len(wav) < 1600:
        return JSONResponse({"error": "empty text or audio too short"}, status_code=400)
    sid = script if script in SCHEME else "devanagari"
    P = posteriors(wav)
    # consensus across decodes (one forward pass, cheap re-argmax at different blank penalties):
    # a syllable is only flagged RED when EVERY decode agrees it was substituted by the SAME token.
    # anything unstable across decodes -> AMBER ("unclear"), never a confident false accusation.
    strict = (mode == "strict")
    hyps = [greedy(P, lam) for lam in (0.0, 3.0, 6.0)]
    ref = _cluster(t)
    opsets = [align_aksharas(ref, _cluster(h)) for h in hyps]
    res = []; n_ok = n_red = n_amber = 0
    for i in range(len(ref)):
        kinds = [opsets[k][i][0] for k in range(len(hyps))]
        subs = [opsets[k][i][1] for k in range(len(hyps)) if opsets[k][i][0] == "sub"]
        any_match = any(k == "match" for k in kinds)
        if strict:                                          # exacting: unanimous-correct or it's flagged
            if all(k == "match" for k in kinds): band = "green"
            elif not any_match: band = "red"                # no decode matched -> clearly off
            else: band = "amber"                            # mixed -> doubtful (counts against)
        else:                                               # liberal: forgiving, only unmistakable errors
            if any_match: band = "green"                    # any decode got it -> credit
            elif all(k == "sub" for k in kinds) and len(set(subs)) == 1: band = "red"
            else: band = "amber"                            # unclear -> set aside (excluded)
        hd = "" if band == "green" else (subs[0] if subs else "")
        res.append({"band": band, "op": band, "heard": from_dev(hd, sid) if hd else ""})
        n_ok += band == "green"; n_red += band == "red"; n_amber += band == "amber"
    denom = (n_ok + n_red + n_amber) if strict else (n_ok + n_red)   # strict: amber counts against
    pct = round(100 * n_ok / denom, 1) if denom else 0.0
    # consented data flywheel: a >=90% chant closely follows the reference, so (audio, reference)
    # is a clean training pair. Save the ORIGINAL uploaded 16k wav + Devanāgarī reference label.
    if pct >= COLLECT_MIN and (n_ok + n_red) >= 3:
        try:
            cid = uuid.uuid4().hex[:12]
            with open(f"{FLYW}/audio/{cid}.wav", "wb") as fh: fh.write(raw)
            with open(f"{FLYW}/log.jsonl", "a") as fh:
                fh.write(json.dumps({"id": cid, "t": int(time.time()), "text": t, "percent": pct,
                                     "script": sid, "mode": mode, "n_ok": n_ok, "n_red": n_red,
                                     "n_amber": n_amber}, ensure_ascii=False) + "\n")
        except Exception:
            pass
    return {"percent": pct, "aksharas": res, "mode": mode, "heard": from_dev(hyps[0], sid),
            "reference": from_dev(t, sid), "n_ok": n_ok, "n_red": n_red, "n_amber": n_amber}

@app.post("/feedback")
async def feedback(audio: UploadFile = File(...), reference: str = Form(""), heard: str = Form(""),
                   script: str = Form(""), kind: str = Form("said_right")):
    """Log a learner's 'I chanted this correctly' override: the audio + reference + what-we-heard,
    so residual false-flags are both harmless (user overrides) and measurable (we count them)."""
    fid = uuid.uuid4().hex[:12]
    try:
        with open(f"{FB}/audio/{fid}.wav", "wb") as f: f.write(await audio.read())
    except Exception:
        pass
    with open(f"{FB}/log.jsonl", "a") as f:
        f.write(json.dumps({"id": fid, "t": int(time.time()), "kind": kind, "script": script,
                            "reference": reference, "heard": heard}, ensure_ascii=False) + "\n")
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def home():
    return open(os.path.join(os.path.dirname(__file__), "practice.html")).read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
