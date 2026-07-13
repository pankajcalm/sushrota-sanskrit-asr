#!/bin/bash
P=/home/ece/BigDisk/Prathosh/ASR/envs/nemo_ai4b/bin/python
D=/home/ece/BigDisk/Prathosh/ASR/bin
mkdir -p $D
$P - << 'PY'
import urllib.request, zipfile, os
urllib.request.urlretrieve('https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip','/tmp/deno.zip')
zipfile.ZipFile('/tmp/deno.zip').extractall('/home/ece/BigDisk/Prathosh/ASR/bin')
os.chmod('/home/ece/BigDisk/Prathosh/ASR/bin/deno',0o755)
PY
if $D/deno --version >/dev/null 2>&1; then
  echo "deno OK: $($D/deno --version | head -1)"
  tmux kill-session -t epgp_dl 2>/dev/null
  tmux new-session -d -s epgp_dl "cd /home/ece/BigDisk/Prathosh/ASR/data/epgp && PATH=$D:\$PATH bash download_all.sh > dl.log 2>&1"
  echo "re-download relaunched (deno in PATH)"
else
  echo "deno install FAILED"
fi
