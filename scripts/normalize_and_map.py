#!/usr/bin/env python3
"""Normalize all raw recitation audio to 16k mono wav + build a file->metadata map.

- Input : data/audio_real/*  (mp3/m4a/mp4/opus/ogg/wav, messy names, spaces, Kannada)
- Output: data/norm16k/<id>.wav  (16k mono pcm_s16le)
          data/norm16k/files_map.json  [{id, orig, wav, work, speaker, skandha,
                                          adhyaya, adhyaya_raw, duration_s, sr, ch, codec}]
Originals are never modified. IDs are safe ASCII slugs; collisions get -2,-3 suffixes.
"""
import os, re, json, subprocess, unicodedata, glob

ROOT = "/home/ece/BigDisk/Prathosh/ASR"
SRC  = f"{ROOT}/data/audio_real"
DST  = f"{ROOT}/data/norm16k"
os.makedirs(DST, exist_ok=True)

AUDIO_EXT = (".mp3", ".m4a", ".mp4", ".opus", ".ogg", ".wav", ".aac", ".flac")

def ffprobe(path):
    def q(entries, stream=False):
        cmd = ["ffprobe", "-v", "error"]
        if stream:
            cmd += ["-select_streams", "a:0"]
        cmd += ["-show_entries", entries, "-of", "csv=p=0", path]
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    dur = q("format=duration")
    codec = q("stream=codec_name", True)
    sr = q("stream=sample_rate", True)
    ch = q("stream=channels", True)
    try: dur = float(dur)
    except: dur = 0.0
    return dur, codec, sr, ch

def detect_work(low):
    if "vayustuti" in low or "vayu" in low:              return "vayustuti"
    if "sahasra" in low:                                  return "vishnu_sahasranama"
    if "tantrasara" in low or "ತಂತ್ರಸಾರ" in low:  return "tantrasara"  # ತಂತ್ರಸಾರ
    if "anuvyakhyana" in low:                             return "anuvyakhyana"
    if "dvadasha" in low or "dwadasha" in low or "stotra" in low or "ದ್ವಾದಶ" in low:  return "dvadasha_stotra"  # ದ್ವಾದಶ
    if "bhagavat" in low or "bahagavata" in low or "skand" in low or "sknd" in low:  return "bhagavatam"
    return "unknown"

def parse_skandha(low):
    m = re.search(r'skan?d?h?a?[ _]*?(\d+)', low)          # skanda_3, skandha_1, skandah_3
    if m: return int(m.group(1))
    m = re.search(r'(\d+)\s*(?:st|nd|rd|th)?\s*skan', low)  # "4th Skandha"
    if m: return int(m.group(1))
    m = re.search(r'skanda?(\d+)', low)                     # Skanda10
    if m: return int(m.group(1))
    return None

def parse_adhyaya(low):
    # capture ranges like 13_16, 5-8, 1-4, 69_to_72, single ints; also "adhyaya 2nd"
    m = re.search(r'adh?a?y?aya?[ _.]*([0-9]+(?:[ _]*(?:to|-|_)[ _]*[0-9]+)?)', low)
    if m:
        raw = re.sub(r'[ _]*to[ _]*', '-', m.group(1)).replace('_', '-').strip('-')
        return raw
    m = re.search(r'(\d+)(?:st|nd|rd|th)?\s*adh?y?aya', low)
    if m: return m.group(1)
    return None

def slug(name):
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode()      # drop Kannada/diacritics
    s = re.sub(r'\.(mp4|mp3|m4a|opus|ogg|wav|aac|flac)', '', s, flags=re.I)  # stacked ext
    s = re.sub(r'[^\w]+', '_', s).strip('_').lower()
    return s or "file"

def main():
    files = sorted(f for f in glob.glob(f"{SRC}/*")
                   if f.lower().endswith(AUDIO_EXT))
    seen, rows = {}, []
    for path in files:
        base = os.path.basename(path)
        low = base.lower()
        work = detect_work(low)
        sk = parse_skandha(low) if work == "bhagavatam" else parse_skandha(low)
        ad = parse_adhyaya(low)
        sid = slug(os.path.splitext(base)[0])
        n = seen.get(sid, 0) + 1; seen[sid] = n
        if n > 1: sid = f"{sid}-{n}"
        wav = f"{DST}/{sid}.wav"
        dur, codec, sr, ch = ffprobe(path)
        # normalize -> 16k mono s16le
        r = subprocess.run(["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", wav],
                           capture_output=True, text=True)
        ok = (r.returncode == 0 and os.path.exists(wav))
        speaker = re.split(r'[_ ]', base, maxsplit=1)[0]
        rows.append(dict(id=sid, orig=base, wav=wav if ok else None, ok=ok,
                         work=work, speaker=speaker, skandha=sk,
                         adhyaya=ad, duration_s=round(dur, 2),
                         sr=sr, ch=ch, codec=codec))
        print(f"[{'ok' if ok else 'FAIL'}] {work:16} sk={str(sk):>4} adh={str(ad):>7} "
              f"{dur/60:5.1f}m  {base[:50]}", flush=True)
    json.dump(rows, open(f"{DST}/files_map.json", "w"), ensure_ascii=False, indent=2)
    # summary
    from collections import defaultdict
    byw = defaultdict(lambda: [0, 0.0])
    for r in rows:
        byw[r["work"]][0] += 1; byw[r["work"]][1] += r["duration_s"]
    nfail = sum(1 for r in rows if not r["ok"])
    print("\n== by work ==")
    for w, (c, d) in sorted(byw.items(), key=lambda x: -x[1][1]):
        print(f"  {w:18} {c:3d} files  {d/3600:5.2f} h")
    print(f"\ntotal {len(rows)} files, {sum(r['duration_s'] for r in rows)/3600:.2f} h, "
          f"{nfail} failed. map -> {DST}/files_map.json")

if __name__ == "__main__":
    main()
