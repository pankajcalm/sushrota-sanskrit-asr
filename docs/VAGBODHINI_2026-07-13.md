# Vāgbodhinī — system & data documentation
**Date:** 2026-07-13
**Author context:** Prof. Prathosh A P, Indian Institute of Science, Bengaluru

Vāgbodhinī (वाग्बोधिनी) is an interactive Sanskrit **chant-practice** tool: a user pastes a
śloka in any script, hears a reference chant synthesised by our **Vāgdhenu** TTS (metre-aware),
chants along, and gets **per-syllable feedback** from our IndicConformer ASR (v5). Every
high-quality attempt is collected (with consent) as clean training data to improve the ASR.

Tagline (header): *"A vṛtta (metre) aware śloka-to-chant text-to-speech system for Sanskrit."*

---

## 1. Services, hosts, ports

| Service | File | Env | GPU | Port | tmux | Purpose |
|---|---|---|---|---|---|---|
| Practice app + UI + ASR/scoring | `/home/ece/BigDisk/Prathosh/ASR/scripts/app_practice.py` | `nemo_ai4b` | 0 | **8010** | `practice` | serves UI, `/detect` `/generate` `/score` `/feedback`; loads v5 (`exp/ft_ctc_v5/ft_ctc_ep20.nemo`) |
| Vāgdhenu TTS microservice | `/home/ece/Prathosh/vagdhenu_serve/tts_api.py` | vagdhenu_serve `venv` | 1 | **8020** | `ttsapi` | `POST /tts {text,meter,seed}→wav`; loads DiT voice + BigVGAN; NO quota (internal) |
| Public tunnel | cloudflared → localhost:8010 | — | — | — | `practice_tunnel` | ephemeral trycloudflare URL |
| (separate) Vāgdhenu public demo | `vagdhenu_serve/server.py` | venv | 0 | 7860 | `vag`/`vagnamed` | the standalone TTS demo (prathosh.in/vagdhenu) — untouched by this tool |

Frontend `practice.html` is served fresh on every `GET /` (edit + no restart needed for UI-only
changes; backend `.py` changes need a `practice` tmux restart).

Restart practice app:
```
cd /home/ece/BigDisk/Prathosh/ASR
tmux kill-session -t practice
tmux new-session -d -s practice \
  "cd /home/ece/BigDisk/Prathosh/ASR && CUDA_VISIBLE_DEVICES=0 envs/nemo_ai4b/bin/python scripts/app_practice.py > logs/practice.log 2>&1"
```
Restart TTS microservice:
```
cd /home/ece/Prathosh/vagdhenu_serve
tmux kill-session -t ttsapi
tmux new-session -d -s ttsapi \
  "cd /home/ece/Prathosh/vagdhenu_serve && CUDA_VISIBLE_DEVICES=1 ./venv/bin/python tts_api.py > tts_api.log 2>&1"
```

Internal design: Devanāgarī is the canonical internal representation. All ASR/scoring/metre/TTS
work in Devanāgarī; the user's script is a presentation layer only.

---

## 2. ★ Data flywheel (consented ASR training data)

**This is the systematically-collected training data.** On every `/score`, when the match is
**≥ `COLLECT_MIN` (env, default 90%)** and there are **≥3 confidently-judged aksharas**, the app
saves the recording + its reference text as a clean training pair.

**Location (persistent BigDisk, 1.8 TB):**
```
/home/ece/BigDisk/Prathosh/ASR/data/practice_flywheel/
    audio/<id>.wav      # ORIGINAL 16 kHz mono WAV the browser recorded (raw; no pre/post-roll pad)
    log.jsonl           # one JSON line per collected chant
```

**`log.jsonl` line schema:**
```json
{"id":"<12-hex>", "t":<unix_seconds>, "text":"<Devanāgarī reference = the LABEL>",
 "percent":<float>, "script":"<input script id>", "mode":"strict|liberal",
 "n_ok":<int>, "n_red":<int>, "n_amber":<int>}
```
- `text` is the **reference** (target) in Devanāgarī — the label. It is a valid transcript because
  at ≥90% match the chant closely follows the reference.
- `audio/<id>.wav` pairs with the log line of the same `id`.

**Why the label is trustworthy:** we do NOT label with the ASR's own output (which would just
reinforce its errors). We label with the *known reference text the user was chanting*, gated by a
high acoustic match — so it's human-produced audio with a human-intended, verified transcript.

**Consent:** a visible footer states — *"By using this service you are consenting for usage of this
data for improving the model."* (Implied consent by use.)

**Second stream — explicit overrides:** when a user taps **"I chanted this correctly"** (overriding
a flag), the audio + reference + what-was-heard are logged to:
```
/home/ece/BigDisk/Prathosh/ASR/data/practice_feedback/{audio/<id>.wav, log.jsonl}   # kind:"said_right"
```
These are human-affirmed correct pairs (also usable, and a signal of ASR false-positives).

**To turn the flywheel into a NeMo training manifest** (when ready to harvest):
```python
import json
rows=[]
for l in open("/home/ece/BigDisk/Prathosh/ASR/data/practice_flywheel/log.jsonl"):
    r=json.loads(l)
    rows.append({"audio_filepath": f"/home/ece/BigDisk/Prathosh/ASR/data/practice_flywheel/audio/{r['id']}.wav",
                 "text": r["text"], "lang": "sa"})   # add "duration" via soundfile if needed
```
**Recommended before training:** review a sample of pairs — a 90% soft-match can still carry a
mislabeled akshara or two. Optionally raise `COLLECT_MIN` (e.g. 95) or filter `n_red==0`.

Current state (2026-07-13): directories exist, **0 clips** (test artifacts cleared); fills as
scholars use the tool.

---

## 3. Rate limiting

- **`DAILY_LIMIT`** (env, default **10**) new ślokas per IP per day — protects the shared TTS GPU.
- Keyed by real visitor IP: `cf-connecting-ip` header (through the Cloudflare tunnel) → `x-forwarded-for` → socket.
- File-backed: `/home/ece/BigDisk/Prathosh/ASR/data/practice_limits.json`, key `ip|YYYY-MM-DD → count`
  (loaded into memory at startup, persisted on increment).
- **Only NEW synthesis counts** — a cached re-generation (same śloka, or toggling script back) is FREE.
- Over limit → HTTP 429 with *"You've reached today's limit of 10 ślokas from this network…"*.

---

## 4. Endpoints (app_practice.py, :8010)

- `GET /` — the UI (practice.html).
- `GET /health` — `{status, device, tts}` (tts = microservice reachable).
- `POST /detect {text, script?}` — detect input script; return `{script, name, echo (round-trip in
  that script for confirmation), devanagari, scripts[]}`.
- `POST /generate {text, script?}` — **streams NDJSON**: `{type:plan,total,remaining}` →
  `{type:progress,done,total}` per render → `{type:done, levels:{pada,ardha,full}, tts_ok}`.
  Each unit: `{i, verse, metre (internal), aksharas:[{text(Devanāgarī), disp(user script), word_end}], wav_url}`.
  Enforces the daily limit (only when new synthesis needed).
- `POST /score {audio, text, script?, mode}` — akshara-level feedback; collects flywheel data on ≥90%.
- `POST /feedback {audio, reference, script, kind}` — logs "I said it right" overrides.

---

## 5. How scoring works (robustness posture)

**Not GOP/forced-alignment** (that produced false positives) — **akshara-level text comparison**
of the free ASR decode vs the reference, made robust by *consensus + abstain*:
- One forward pass → 3 greedy decodes at blank penalties {0,3,6} (cheap re-argmax).
- Both sides canonicalised (`dedup` doubled marks; word-final म् ≡ anusvāra ं; spacing).
- Align each decode's aksharas to the reference (edit distance).
- Per reference akshara, by mode:
  - **Be-strict (default):** green only if ALL decodes match; red if NO decode matches; else amber. `% = green/(green+red+amber)`.
  - **Be-liberal:** green if ANY decode matches; red only if ALL agree on the SAME wrong syllable; else amber. `% = green/(green+red)` (amber excluded).
- **amber = "unclear"** — the tool abstains rather than confidently false-accuse.
- UI: green / red (hover: "heard: X") / amber; plus **playback of your recording vs reference**,
  and the **"I chanted this correctly"** override.

Design principle: *never confidently wrong, and always verifiable.* The ASR (~6% CER) is imperfect,
so the tool points and the user's ear judges.

---

## 6. Metre (vṛtta) detection & levels

- Input (any script) → Devanāgarī → split verses on ॥ / blank line → per verse, `detect_pada_len`
  identifies the metre by matching **pāda 1** to a known L/G signature (tolerant of minor liberties
  in later pādas), else anuṣṭubh by count (8/16/32), else generic identical-quarters, else prose.
- Learning levels generated (each a SEPARATE Vāgdhenu render, never audio-joined): **pāda (¼)**,
  **ardha (½)**, **full**. A view toggle switches between the pre-generated levels.
- The **detected metre is used internally only** (TTS reference-bank voice + pāda boundaries) and is
  **NOT displayed** (a wrong label would erode trust). Sample-card metre labels are curated.

---

## 7. UI features
- **Any-script input** (Devanāgarī/Kannada/Telugu/Tamil/Malayalam/Bengali/Grantha/… + Roman IAST/HK/
  ITRANS/SLP1), auto-detected with a **confirmation echo** and a **script-override picker**; entire
  UI renders in the user's script.
- **Sample ślokas** (5, hide on select): Kṛṣṇa (anuṣṭubh/Devanāgarī), Hari maṅgala
  (śārdūlavikrīḍita/Kannada), Narasiṃha (mālinī/Devanāgarī), Guru stotra (anuṣṭubh/Kannada),
  Sarasvatī (anuṣṭubh/Telugu). Pre-warmed in the TTS cache.
- **Recording:** raw audio (echoCancellation/noiseSuppression/AGC OFF), 3-2-1 countdown + **beep**
  at "GO" (mic warm-up prevents onset clipping), pulsing red dot + timer + live level meter.
- **Reference playback speed** 0.75× / 1× / 1.25× (client-side `playbackRate`, pitch preserved).
- **Generation progress bar** (fills as each of the 7 units renders).

---

## 8. Key file inventory
```
SERVER /home/ece/BigDisk/Prathosh/ASR/
  scripts/app_practice.py     backend (:8010) — UI, ASR, scoring, flywheel, limits
  scripts/practice.html       frontend (served at GET /)
  scripts/prewarm.py          pre-warm sample ślokas into TTS cache
  data/tts_cache/             synthesised reference wavs (sha1(text|metre|seed))
  data/practice_flywheel/     ★ consented training data (audio/ + log.jsonl)
  data/practice_feedback/     "I said it right" overrides
  data/practice_limits.json   per-IP daily counters
  exp/ft_ctc_v5/ft_ctc_ep20.nemo   the ASR model (v5)
SERVER /home/ece/Prathosh/vagdhenu_serve/
  tts_api.py                  TTS microservice (:8020)
  weights/voice_steer.pt      Vāgdhenu DiT voice
MAC /Users/prathosh/ASR/outputs/   local copies of the above scripts
```

---

## 9. Known limitations / next steps
- ASR v5 ~6% CER — text-comparison inherits its errors (mitigated by consensus/abstain + override).
- Flywheel labels are soft-matched (≥90%); review before training; consider raising the threshold.
- Public URL is currently the **ephemeral** trycloudflare tunnel — make a durable named tunnel
  (like Vāgdhenu's) for a stable address.
- Reference "slow" is playback-rate (time-stretch); a native slow-tempo TTS render is possible via
  Vāgdhenu's `sps`/`speed` param if higher quality is wanted (at extra GPU/quota cost).
