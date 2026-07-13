# Su-śrotā — Scholar-grade Sanskrit ASR & chant practice

Sanskrit speech recognition tuned for **śāstric / recitational** Sanskrit, plus **Vāgbodhinī**,
an interactive chant-practice tool built on it. This repository documents the full set of
experiments — including the ones that **did not** work, which were as instructive as the ones
that did.

- **ASR model (finetuned):** IndicConformer-CTC (Sanskrit slice), "v5" — released on Hugging Face.
- **Chant-practice app:** Vāgbodhinī — paste a śloka (any script), hear a metre-aware reference
  chant (Vāgdhenu TTS), chant along, get per-syllable feedback.
- **Author:** Prof. Prathosh A P, Indian Institute of Science, Bengaluru.

---

## 1. Motivation

Off-the-shelf Sanskrit ASR is trained on conversational IndicVoices-style data and degrades badly
on **recitation and śāstra** — dense compounds, sandhi, retroflex/aspirate contrasts, pitch, and
long metrical utterances. The goal here was a model and tooling good enough for **scholars**:
accurate on chant and prose, and useful as a *practice aid* rather than a transcription toy.

Two things shaped everything:
1. **Segmentation dominates Sanskrit WER.** Word boundaries in Sanskrit are largely a writing
   convention (sandhi fuses words). Standard WER punishes boundary disagreements that are not
   errors. We therefore report a **sandhi-normalized WER (SN-WER)** alongside CER/WER.
2. **The acoustic model saturates.** Past a point, adding data stops moving CER; the leverage moves
   to how you *use* the model (known-reference tasks, decoding, tooling), not how you train it.

---

## 2. Data & preprocessing

- **Recitation + śāstra prose** corpora (Bhāgavata chant, Vedānta prose), a TTS-speaker slice, and
  a **multi-scholar annotation** set collected via a purpose-built web portal.
- **Content-only normalization:** NFC, keep Devanāgarī letters + combining marks, drop digits,
  daṇḍa, avagraha, oṁ, and Vedic accent marks; collapse whitespace.
- **Metrics:**
  - **CER** — character error rate on space-stripped text (the primary, boundary-agnostic metric).
  - **WER** — whitespace-token error rate (segmentation-sensitive).
  - **SN-WER** — *sandhi-normalized* WER: strip spaces on both sides, char-align, a reference word
    counts correct iff all its characters are recovered. Reported as a band `lo..hi`.

Representative v5 numbers (held-out):

| domain | CER | WER | SN-WER |
|---|---|---|---|
| e-PG lecture (gold) | 3.6% | 13.0% | ~9–10% |
| Bhāgavata chant | 6.0% | 46.4% | ~22–26% |
| Vedānta prose | 7.3% | 30.8% | ~15–19% |

The gap between WER and SN-WER quantifies point (1): roughly **half of Sanskrit "WER" is spacing.**

---

## 3. The model (v5)

- **Architecture:** `EncDecHybridRNNTCTCBPEModel` (IndicConformer, ~129 M params). We use the **CTC
  head** on the Sanskrit token slice (`cols = [BLANK] + range(4096,4352)`, re-`log_softmax`), greedy
  decode.
- **Training data (v5):** recitation + disk-prose + TTS-speaker slice.
- **Result:** the shipped model — chant CER **6.0%**. Weights on Hugging Face (see §7).

---

## 4. Experiments — what we tried

A compressed log of the campaign. **Bold = shipped / kept.**

| # | Experiment | Idea | Outcome |
|---|---|---|---|
| 1 | **v5 finetune (CTC)** | finetune IndicConformer on recitation+prose+TTS | **shipped; chant CER 6.0%** |
| 2 | **SN-WER metric** | strip sandhi/spacing before scoring | **adopted; showed ½ of WER is spacing** |
| 3 | v8 pseudo-labels | self-train on high-confidence v5 outputs | ✗ confidence doesn't separate right/wrong (wrong words median conf 0.93); label noise → no gain |
| 4 | **Hard-negative annotation** | pick hardest clips by v5-vs-Whisper disagreement for scholars to label | **adopted for the annotation drive** |
| 5 | **Multi-scholar portal** | concurrency-safe claim-queue annotation web tool | **shipped; used to collect clean labels** |
| 6 | SSL pretraining | wav2vec2-style contrastive on 38 h of Sanskrit śāstric audio; freeze frontend+lower layers | ✗ SSL-init finetune (v9b) < clean-label finetune (v9a) |
| 7 | Semi-supervised pseudo-labels | v5/v9 teacher → confidence-gated pseudo-labels | ✗ confirmed to hurt (3 independent tests) |
| 8 | v9 finetunes (v9a / v9-v2 / v9-v3) | add 846 clean scholar labels (47 speakers) + gold, honest held-out split | ✗ CER flat vs v5; gold **WER worse** (segmentation drift). First "−28%" was a favorable-split artifact |
| 9 | Rule-based sandhi segmenter | DP split of merged tokens to fix WER | ✗ over-shatters (Sanskrit tiles into everything); WER worse |
| 10 | ByT5-Sanskrit post-corrector | byte-level seq2seq to fix ASR output | ✗ over-corrects rare terms (prose CER 7→22); segmentation concept helped chant WER 46→42 only |
| 11 | Blank-penalty / r-recall sweep | penalize CTC blank to recover dropped onsets/repha | partial; diagnostic — CTC blank bias explains onset/short-phone deletions |
| 12 | Onset & CER-distribution analysis | per-position and per-clip error structure | diagnostic — first 2–3 words dropped (encoder ramp-up); e-PG tail is speaker-driven & bimodal |
| 13 | Baselines | Whisper-sa, wav2vec2 finetunes | v5 (IndicConformer-CTC) remained the best on chant/prose |
| 14 | **GOP forced alignment** | Goodness-of-Pronunciation for chant scoring | validated (AUC 0.97–0.99) but too false-positive-prone for a tutor → superseded |
| 15 | **Vāgbodhinī chant tool** | exploit the *known reference* text: verify, don't transcribe | **shipped** (see §6) |

---

## 5. Key findings

1. **Clean human labels are the only training lever that helps at all — and even they don't beat v5
   honestly.** With a properly held-out eval, adding 846 scholar labels left CER flat and hurt gold
   WER (the annotators' spacing conventions drifted the model's word segmentation).
2. **The acoustic model is saturated (~6% CER on clean chant).** SSL, pseudo-labels, and more
   finetuning did not move it.
3. **Inference-side text fixes (rule segmenter, ByT5) don't transfer** — Sanskrit's productive
   sandhi makes naive segmentation/correction over-fire.
4. **The real win is task reformulation.** For chant practice we *know the target text*, so the
   problem is **verification, not recognition** — which sidesteps the ~6% CER ceiling entirely.
5. **CTC is spiky:** a correctly-heard syllable fires at one frame with blank ("continuation")
   around it. Any per-frame confidence measure must treat blank as neutral, not as evidence against
   the target — a subtlety that caused (and, once understood, fixed) a class of false negatives.

---

## 6. Vāgbodhinī — the chant-practice tool

`vagbodhini/` — a standalone web app.

- **Any-script input** (Devanāgarī/Kannada/Telugu/Tamil/Malayalam/Bengali/Grantha + Roman
  IAST/HK/ITRANS/SLP1) → auto-detected, confirmed by a round-trip echo, UI rendered in the user's
  script. Devanāgarī is the internal canonical form.
- **Metre-aware (vṛtta) pāda splitting:** identify the metre by matching pāda 1 to a known
  laghu/guru signature (tolerant of real metrical liberties), then split into **pāda (¼) / ardha
  (½) / full** learning units. Each is rendered *separately* by the **Vāgdhenu** metre-conditioned
  TTS (not stitched).
- **Feedback = akshara-level text comparison** of the ASR decode vs the reference, made robust by
  **decode-consensus + abstain**: only flag a syllable red when several decodes agree; when the
  model is unsure, say *"unclear"* rather than falsely accuse. **Be-strict / Be-liberal** modes let
  the user choose the precision/recall point.
- **Playback + override:** hear your recording vs the reference; "I chanted this correctly" overrides
  and logs the case.
- **Consented data flywheel:** attempts matching the reference ≥90% are saved as clean
  (audio, reference-text) training pairs — the tool improves the very ASR underneath it.

See `docs/VAGBODHINI_2026-07-13.md` for the full system spec (services, ports, endpoints, data
format).

---

## 7. Model weights

The finetuned **v5** Sanskrit ASR (IndicConformer-CTC) is released on Hugging Face:

> **`prathoshap/sushrota-sanskrit-asr`** *(see the HF model card for usage)*

The metre-aware TTS used for reference chants, **Vāgdhenu**, is separately at
`prathoshap/vagdhenu`.

---

## 8. Repository layout

```
vagbodhini/      the chant-practice app (FastAPI backend + single-file UI + TTS microservice)
scripts/         all experiment / training / evaluation / analysis scripts (§4)
docs/            system documentation
README.md        this report
```

Notes:
- Corpora, audio, and model checkpoints are **not** in the repo (weights → Hugging Face).
- Scripts assume the training/eval data layout on the lab server and are provided for
  documentation and reproducibility of method, not turnkey execution.

---

## 9. Citation

If you use this work, please cite:

> Prathosh A P, *Su-śrotā: Scholar-grade Sanskrit ASR and metre-aware chant practice*, Indian
> Institute of Science, Bengaluru, 2026.
