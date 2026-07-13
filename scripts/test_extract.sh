#!/bin/bash
cd /home/ece/BigDisk/Prathosh/ASR/data/epgp
Y=/home/ece/BigDisk/Prathosh/ASR/envs/nemo_ai4b/bin/yt-dlp
U=$(head -1 urls.txt)
rm -f test_done test16k.wav slice.wav test_raw.*
$Y -x --audio-format wav --no-warnings -o "test_raw.%(ext)s" "$U" 2>test_dl.err
ffmpeg -y -i test_raw.wav -ac 1 -ar 16000 test16k.wav 2>/dev/null
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 test16k.wav 2>/dev/null)
ffmpeg -y -ss 120 -t 45 -i test16k.wav slice.wav 2>/dev/null
echo "url: $U"
echo "dur: ${DUR}s   size: $(ls -la test16k.wav 2>/dev/null | awk '{print $5}')"
echo "--- model on a 45s slice (t=120) ---"
curl -s -X POST http://127.0.0.1:8000/transcribe -F "audio=@slice.wav" -F "interim=true" 2>/dev/null | head -c 700
echo; echo TESTDONE > test_done
