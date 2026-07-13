#!/usr/bin/env python3
"""Confidence-gated utterance builder for Bhagavatam recitation audio.

Per file: silence-split -> CTC transcribe -> anchor hypotheses to the canonical
adhyaya text -> keep segments whose hypothesis matches the reference (low CER),
labelling each with the CANONICAL text (not the ASR output). Divergences (reciter
mistakes, commentary) fail the match and are dropped.

Run in envs/nemo_ai4b. Usage:
  python segment_align.py --ids Sri_Skanda_3_adhyaya_29     # smoke one file (by id prefix)
  python segment_align.py --work bhagavatam --exclude-sk 10 # full train pool
"""
import os, re, json, glob, subprocess, unicodedata, argparse, difflib
import numpy as np, soundfile as sf

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
MAP  = f"{ROOT}/data/norm16k/files_map.json"
TEXTS= f"{ROOT}/data/texts/bhagavatam"
OUT  = f"{ROOT}/data/utts"
MODEL=("/home/ece/.cache/torch/NeMo/NeMo_2.7.3/hf_hub_cache/ai4bharat/"
       "indicconformer_stt_sa_hybrid_ctc_rnnt_large/c82246cc7136e7f1be8df3090e3a07d9/"
       "indicconformer_stt_sa_hybrid_rnnt_large.nemo")

# ---------- text ----------
KEEP = re.compile(r'[^ऀ-ॿ\s]')
DROP = re.compile(r'[०-९।॥ऽॐ]')     # digits, dandas, avagraha, om
def norm(s):
    s = unicodedata.normalize('NFC', s)
    s = DROP.sub(' ', s); s = KEEP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def adh_set(spec):
    """'13-16' -> {13,14,15,16}; '17' -> {17}"""
    out = set()
    for part in str(spec).split('-'):
        part = part.strip()
        if part.isdigit(): out.add(int(part))
    if '-' in str(spec):
        a, b = str(spec).split('-')[0], str(spec).split('-')[-1]
        if a.isdigit() and b.isdigit(): out = set(range(int(a), int(b)+1))
    return out

def canonical_text(skandha, adhyaya_spec):
    """Ordered normalized verse strings for skandha + adhyaya(range)."""
    p = f"{TEXTS}/skandha_{skandha}.json"
    if not os.path.exists(p): return []
    d = json.load(open(p))
    want = adh_set(adhyaya_spec)
    items = []
    for v in d["content"].values():
        for e in v:
            a = e.get("adhyaya")
            if a in want:
                verse = e.get("verse")
                vn = verse if isinstance(verse, int) else 10**6
                items.append((a, vn, " ".join(e.get("text", []))))
    items.sort(key=lambda x: (x[0], x[1]))
    return [norm(t) for a, vn, t in items if norm(t)]

# ---------- silence segmentation ----------
def silences(wav, noise="-30dB", d=0.35):
    r = subprocess.run(["ffmpeg", "-i", wav, "-af",
                        f"silencedetect=noise={noise}:d={d}", "-f", "null", "-"],
                       capture_output=True, text=True)
    st, se = [], []
    for line in r.stderr.splitlines():
        m = re.search(r'silence_start:\s*([\d.]+)', line)
        if m: st.append(float(m.group(1)))
        m = re.search(r'silence_end:\s*([\d.]+)', line)
        if m: se.append(float(m.group(1)))
    return list(zip(st, se))

def segments(wav, dur, tgt_min=4.0, tgt_max=16.0):
    sils = silences(wav)
    cuts = [0.0] + [ (s+e)/2 for s, e in sils ] + [dur]
    cuts = sorted(set(round(c, 3) for c in cuts if 0 <= c <= dur))
    raw = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1]-cuts[i] > 0.2]
    # greedy-merge to reach tgt_min
    merged, cur = [], None
    for s, e in raw:
        if cur is None: cur = [s, e]
        elif cur[1]-cur[0] < tgt_min: cur[1] = e
        else: merged.append(tuple(cur)); cur = [s, e]
    if cur: merged.append(tuple(cur))
    # split any > tgt_max evenly
    final = []
    for s, e in merged:
        L = e - s
        if L <= tgt_max: final.append((s, e)); continue
        n = int(np.ceil(L / tgt_max)); step = L / n
        final += [(s+i*step, s+(i+1)*step) for i in range(n)]
    return final

# ---------- reference anchoring ----------
def anchor(seg_hyps, ref_text):
    """Char-level global anchoring: map each segment's hyp chars to a ref char span
    via one global alignment (robust to sandhi/word-boundary differences). Returns
    (ref_span, cer) per segment."""
    hyp_chars, owner = [], []
    for si, h in enumerate(seg_hyps):
        for ch in h: hyp_chars.append(ch); owner.append(si)
        hyp_chars.append(' '); owner.append(si)          # seg separator
    ref_chars = list(ref_text)
    sm = difflib.SequenceMatcher(a=hyp_chars, b=ref_chars, autojunk=False)
    h2r = {}
    for a, b, size in sm.get_matching_blocks():
        for k in range(size): h2r[a+k] = b+k
    byseg = {}
    for i, si in enumerate(owner):
        if i in h2r: byseg.setdefault(si, []).append(h2r[i])
    out = []
    for si in range(len(seg_hyps)):
        rl = byseg.get(si)
        if not rl: out.append(("", 1.0)); continue
        r0, r1 = min(rl), max(rl)
        while r0 > 0 and ref_chars[r0-1] != ' ': r0 -= 1
        while r1 < len(ref_chars)-1 and ref_chars[r1+1] != ' ': r1 += 1
        ref_span = ''.join(ref_chars[r0:r1+1]).strip()
        out.append((ref_span, cer(seg_hyps[si], ref_span)))
    return out

def cer(a, b):
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if not b: return 1.0
    n, m = len(a), len(b)
    if n == 0: return 1.0
    prev = list(range(m+1))
    for i in range(1, n+1):
        curr = [i] + [0]*m
        for j in range(1, m+1):
            curr[j] = min(prev[j]+1, curr[j-1]+1, prev[j-1]+(a[i-1] != b[j-1]))
        prev = curr
    return prev[m] / max(1, m)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None, help="comma id-prefix filter (smoke test)")
    ap.add_argument("--work", default="bhagavatam")
    ap.add_argument("--exclude-sk", type=int, default=None)
    ap.add_argument("--keep-cer", type=float, default=0.25)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--adh", default=None, help="override adhyaya for selected files (diagnostic)")
    args = ap.parse_args()

    rows = json.load(open(MAP))
    rows = [r for r in rows if r["ok"] and r["work"] == args.work]
    if args.exclude_sk is not None:
        rows = [r for r in rows if r["skandha"] != args.exclude_sk]
    if args.ids:
        pref = tuple(args.ids.split(","))
        rows = [r for r in rows if r["id"].startswith(pref)]
    print(f"processing {len(rows)} files", flush=True)

    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL, map_location="cuda")
    m.eval()
    try: m.cur_decoder = "ctc"
    except Exception: pass
    m.change_decoding_strategy(decoder_type="ctc")

    os.makedirs(OUT, exist_ok=True)
    manifest, report, allsegs = [], [], []
    CEIL = 0.35        # retain wavs up to this CER so thresholds can be re-tuned offline
    for r in rows:
        adh = args.adh if args.adh else r["adhyaya"]
        ref_text = " ".join(canonical_text(r["skandha"], adh))
        if not ref_text:
            report.append((r["id"], 0, 0, 0.0, "NO_REF_TEXT")); continue
        wav = r["wav"]; dur = r["duration_s"]
        segs = segments(wav, dur)
        audio, sr = sf.read(wav)
        segdir = f"{OUT}/{r['id']}"; os.makedirs(segdir, exist_ok=True)
        paths = []
        for i, (s, e) in enumerate(segs):
            p = f"{segdir}/seg_{i:04d}.wav"
            sf.write(p, audio[int(s*sr):int(e*sr)], sr); paths.append(p)
        res = m.transcribe(paths, batch_size=16, language_id="sa")
        if isinstance(res, tuple): res = res[0]        # hybrid returns (ctc, rnnt)
        hyps = [norm(h if isinstance(h, str) else getattr(h, "text", str(h)))
                for h in res]
        anchored = anchor(hyps, ref_text)
        if args.debug:
            print(f"  [ref {len(ref_text)} chars] first segments:", flush=True)
            for i in range(min(6, len(hyps))):
                print(f"    seg{i} cer={anchored[i][1]:.2f}\n      HYP: {hyps[i][:70]}\n"
                      f"      REF: {anchored[i][0][:70]}", flush=True)
        kept = 0; kept_dur = 0.0
        for i, ((s, e), (ref_span, c)) in enumerate(zip(segs, anchored)):
            dur_i = round(e-s, 2)
            sane = bool(ref_span) and (e-s) >= 1.0
            allsegs.append(dict(src=r["id"], seg=i, dur=dur_i, cer_ref=round(c, 3),
                                speaker=r["speaker"], skandha=r["skandha"],
                                adhyaya=r["adhyaya"], work=r["work"]))
            if sane and c <= args.keep_cer:
                manifest.append(dict(audio_filepath=paths[i], text=ref_span,
                                     duration=dur_i, speaker=r["speaker"],
                                     skandha=r["skandha"], adhyaya=r["adhyaya"],
                                     work=r["work"], src=r["id"], cer_ref=round(c, 3)))
                kept += 1; kept_dur += (e-s)
            elif not (sane and c <= CEIL):
                os.remove(paths[i])        # only discard clearly-bad wavs
        report.append((r["id"], len(segs), kept, round(kept_dur/60, 2),
                       f"{100*kept_dur/max(1,dur):.0f}% of {dur/60:.1f}m"))
        print(f"  {r['id'][:42]:42} segs={len(segs):3d} kept={kept:3d} "
              f"{kept_dur/60:5.1f}m  {report[-1][4]}", flush=True)

    tag = args.ids.replace(",", "_") if args.ids else f"{args.work}_ex{args.exclude_sk}"
    mpath = f"{OUT}/manifest_{tag}.jsonl"
    with open(mpath, "w") as f:
        for u in manifest: f.write(json.dumps(u, ensure_ascii=False)+"\n")
    with open(f"{OUT}/allsegs_{tag}.jsonl", "w") as f:
        for a in allsegs: f.write(json.dumps(a, ensure_ascii=False)+"\n")
    tot_kept = sum(u["duration"] for u in manifest)
    # yield-per-threshold table
    print("\n== yield vs CER threshold ==", flush=True)
    for th in (0.10, 0.15, 0.20, 0.25, 0.30):
        h = sum(a["dur"] for a in allsegs if a["dur"] >= 1.0 and a["cer_ref"] <= th)/3600
        print(f"  CER<={th:.2f}:  {h:.2f} h", flush=True)
    print(f"\nKEPT(@{args.keep_cer}) {len(manifest)} utts, {tot_kept/3600:.2f} h -> {mpath}", flush=True)

if __name__ == "__main__":
    main()
