# Local Vāgbodhinī quickstart

The practice UI and Sushrota ASR/scoring service run from this repository. Reference-audio
generation is optional and requires the separate Vāgdhenu repository and a CUDA GPU.

## Windows PowerShell

Use Python 3.11. The commands below invoke the virtual environment directly, so activation is
not required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# NVIDIA GPU build (also falls back to CPU when CUDA is unavailable)
.\.venv\Scripts\python.exe -m pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

.\.venv\Scripts\python.exe -m pip install "nemo_toolkit[asr]==2.7.3" fastapi uvicorn python-multipart numpy soundfile requests indic-transliteration huggingface-hub

New-Item -ItemType Directory -Force models
.\.venv\Scripts\hf.exe download prathoshap/sushrota-sanskrit-asr sushrota_sanskrit_asr_v5.nemo --local-dir models

.\.venv\Scripts\python.exe vagbodhini\app_practice.py
```

Open <http://localhost:8010>. Model startup can take about a minute on CPU. The NeMo messages
about missing training, validation, and test manifests are harmless for inference.

To activate the environment instead of using its executables directly:

```powershell
.\.venv\Scripts\Activate.ps1
```

PowerShell uses the backtick for line continuation, not the Bash backslash. Keeping the commands
on one line avoids that difference.

## Using the practice app

1. Paste a Sanskrit verse in Devanāgarī, another supported Indic script, IAST, HK, ITRANS, or SLP1.
2. Confirm the auto-detected script and select **Prepare practice**.
3. Choose pāda, ardha, or full-verse practice.
4. Select **Chant**, allow microphone access, recite after the countdown, and select **Stop**.
5. Review the per-akṣara colors and strict/liberal score.

The app remains usable when `/health` reports `"tts": false`: reference playback is unavailable,
but segmentation, microphone recording, ASR scoring, and feedback still work.

## Optional Vāgdhenu reference audio

`vagbodhini/tts_api.py` is an adapter for the separate
[Vāgdhenu repository](https://github.com/prathoshap/vagdhenu). Vāgdhenu currently requires
Python 3.10 and a CUDA 12.1 GPU. Follow that repository's `scripts/setup.sh`, activate its Python
environment, then start the adapter from this repository:

```powershell
$env:VAGDHENU_HOME = "C:\path\to\vagdhenu"
python vagbodhini\tts_api.py
```

The practice server checks `http://localhost:8020` by default; override it with `TTS_URL`.

Useful checks:

```powershell
Invoke-RestMethod http://localhost:8010/health
Invoke-RestMethod http://localhost:8020/health  # only when optional TTS is running
```
