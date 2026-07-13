#!/bin/bash
cd /home/ece/BigDisk/Prathosh/ASR/data/epgp
Y=/home/ece/BigDisk/Prathosh/ASR/envs/nemo_ai4b/bin/yt-dlp
rm -f scope_done meta.txt meta.err
$Y --skip-download --ignore-errors --no-warnings --print "%(duration)s|%(channel)s|%(title).90s" -a urls.txt > meta.txt 2> meta.err
echo DONE > scope_done
