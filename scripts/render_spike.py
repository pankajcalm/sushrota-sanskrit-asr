#!/usr/bin/env python3
"""Vagdhenu render spike: does render_one sound good on a single pada vs ardha vs full?
Mirrors vagdhenu_serve/server.py's exact Renderer setup. Renders the Kṛṣṇa anuṣṭubh at all
three granularities, times each, saves wavs to ASR/data/tts_spike/ for listening."""
import os, sys, time
HERE = "/home/ece/Prathosh/vagdhenu_serve"
SRC = os.path.join(HERE, "src"); sys.path.insert(0, SRC)
import soundfile as sf
from render_core import Renderer

BANK = os.path.join(SRC, "reference_bank", "bank.json")
VOCAB = os.path.join(SRC, "reference_bank", "vocab.txt")
VOICE = os.path.join(HERE, "weights", "voice_steer.pt")
VOC = "/home/ece/Prathosh/CHAMPION_2026-06-11/voc_bigvgan_EMA_2026-06-11.pth"
NFE = 32
OUT = "/home/ece/BigDisk/Prathosh/ASR/data/tts_spike"; os.makedirs(OUT, exist_ok=True)

print("[boot] loading Renderer on", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
t0 = time.time()
R = Renderer(VOICE, VOC, BANK, device="cuda", vocab_file=VOCAB, nfe=NFE)
print("[boot] warm in %.1fs" % (time.time() - t0), flush=True)

METER = "anuṣṭubh"
FULL = "वसुदेवसुतं देवं कंसचाणूरमर्दनम् ।\nदेवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम् ॥"
PADAS = ["वसुदेवसुतं देवं", "कंसचाणूरमर्दनम्", "देवकीपरमानन्दं", "कृष्णं वन्दे जगद्गुरुम्"]
ARDHAS = ["वसुदेवसुतं देवं कंसचाणूरमर्दनम्", "देवकीपरमानन्दं कृष्णं वन्दे जगद्गुरुम्"]

def render(tag, text):
    t = time.time(); sr, audio = R.render_one(text, METER, seed=60)
    dt = time.time() - t; dur = len(audio) / sr
    p = os.path.join(OUT, tag + ".wav"); sf.write(p, audio, sr)
    print("  %-10s %.2fs render  %.2fs audio  sr=%d  -> %s" % (tag, dt, dur, sr, tag + ".wav"), flush=True)

print("\n== FULL =="); render("full", FULL)
print("== ARDHA =="); [render("ardha%d" % (i + 1), a) for i, a in enumerate(ARDHAS)]
print("== PADA =="); [render("pada%d" % (i + 1), p) for i, p in enumerate(PADAS)]
print("\nSPIKE DONE", flush=True)
