#!/bin/bash
cd /home/ece/BigDisk/Prathosh/ASR/data/epgp
mkdir -p wav16k
Y=/home/ece/BigDisk/Prathosh/ASR/envs/nemo_ai4b/bin/yt-dlp
rm -f dl_done
n=0; tot=$(wc -l < urls.txt)
while read U; do
  ID=$(echo "$U" | sed 's/.*v=//')
  n=$((n+1))
  [ -f wav16k/$ID.wav ] && { echo "[$n/$tot] $ID exists, skip"; continue; }
  $Y -x --audio-format wav --no-warnings -o "raw_$ID.%(ext)s" "$U" 2>>dl.err
  if [ -f raw_$ID.wav ]; then
    ffmpeg -y -i raw_$ID.wav -ac 1 -ar 16000 wav16k/$ID.wav 2>/dev/null
    rm -f raw_$ID.wav
    echo "[$n/$tot] $ID OK"
  else
    echo "[$n/$tot] $ID FAILED"
  fi
done < urls.txt
echo "downloaded $(ls wav16k/*.wav 2>/dev/null | wc -l) files" 
echo ALLDONE > dl_done
