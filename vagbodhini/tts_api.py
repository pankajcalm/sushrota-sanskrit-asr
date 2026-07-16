#!/usr/bin/env python3
"""Vagdhenu internal TTS microservice for the chanting-practice tool.

Mirrors vagdhenu_serve/server.py's Renderer setup but exposes a plain JSON/wav API with NO
Gradio and NO IP quota (internal use only). Loads the model once; renders whatever text chunk
it's given (a pāda, an ardha, or a full śloka). Meter-conditioned via the reference bank; an
invalid/unknown meter hint falls back to the model's own detector, then a neutral default.

Run:  cd /home/ece/Prathosh/vagdhenu_serve && CUDA_VISIBLE_DEVICES=1 ./venv/bin/python tts_api.py
"""
import os, sys, io, json, time
THIS_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.abspath(os.environ.get(
    "VAGDHENU_HOME", os.path.join(os.path.dirname(THIS_REPO), "vagdhenu")
))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(HERE, "BigVGAN"))
import numpy as np, soundfile as sf
from fastapi import FastAPI, Form
from fastapi.responses import Response, JSONResponse
try:
    from render_core import Renderer, detect_meter_key
except ModuleNotFoundError as e:
    if e.name == "render_core":
        raise SystemExit(
            f"Vāgdhenu source not found under {HERE}. Clone and set up "
            "https://github.com/prathoshap/vagdhenu, then set VAGDHENU_HOME to that directory."
        ) from e
    raise

BANK = os.environ.get("VAGDHENU_BANK", os.path.join(SRC, "reference_bank", "bank.json"))
VOCAB = os.environ.get("VAGDHENU_VOCAB", os.path.join(SRC, "reference_bank", "vocab.txt"))
VOICE = os.environ.get("VAGDHENU_VOICE", os.path.join(HERE, "models", "voice_steer_ema_2026-06-17.pt"))
VOC = os.environ.get("VAGDHENU_VOCODER", os.path.join(HERE, "models", "voc_bigvgan_EMA_2026-06-11.pth"))
NFE = int(os.environ.get("VAGDHENU_NFE", "32"))
PORT = int(os.environ.get("TTS_PORT", "8020"))

_missing = [p for p in (BANK, VOCAB, VOICE, VOC) if not os.path.isfile(p)]
if _missing:
    raise SystemExit("Vāgdhenu setup is incomplete; missing:\n  " + "\n  ".join(_missing))

print(f"[boot] loading Renderer (nfe={NFE}) …", flush=True)
t0 = time.time()
R = Renderer(VOICE, VOC, BANK, device="cuda", vocab_file=VOCAB, nfe=NFE)
_bank = json.load(open(BANK, encoding="utf-8"))
VALID = {k for k, v in _bank.items() if not k.startswith("_") and isinstance(v, dict) and "wav" in v}
FALLBACK = "vasantatilakā" if "vasantatilakā" in VALID else ("anuṣṭubh" if "anuṣṭubh" in VALID else next(iter(VALID)))
print(f"[boot] warm in {time.time()-t0:.1f}s | {len(VALID)} meters | fallback={FALLBACK}", flush=True)

def resolve_meter(hint, text):
    if hint in VALID: return hint
    try:
        d = detect_meter_key(text)
        if d in VALID: return d
    except Exception:
        pass
    return FALLBACK

app = FastAPI()

@app.get("/health")
def health(): return {"status": "ok", "meters": sorted(VALID)}

@app.post("/tts")
def tts(text: str = Form(...), meter: str = Form(""), seed: int = Form(60)):
    text = (text or "").strip()
    if not text: return JSONResponse({"error": "empty text"}, status_code=400)
    m = resolve_meter(meter, text)
    try:
        sr, audio = R.render_one(text, m, seed=int(seed))
    except Exception as e:
        return JSONResponse({"error": f"render failed: {e}", "meter": m}, status_code=500)
    buf = io.BytesIO(); sf.write(buf, np.asarray(audio, dtype="float32"), sr, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")  # meter name is non-latin-1; no header

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
