"""
config.py
=========
Single source of truth for every tunable constant.
Never scatter magic numbers across files again.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).parent.resolve()
DATA_FILE          = BASE_DIR / "data.json"
NOTES_DIR          = BASE_DIR / "notes"
RECORDINGS_DIR     = BASE_DIR / "call_recordings"
STT_WAV_PATH       = BASE_DIR / "input_vad.wav"
FAN_FILE           = BASE_DIR / "fan.mp3"
HONK_FILE          = BASE_DIR / "honk.mp3"

# ── Audio — STT ───────────────────────────────────────────────────────────────
STT_SAMPLE_RATE      = 16_000
STT_CHANNELS         = 1
STT_CHUNK_MS         = 30
STT_ENERGY_THRESHOLD = 300
STT_SILENCE_DURATION = 0.5     # seconds of trailing silence before cut-off
STT_MIN_SPEECH_SECS  = 0.3
STT_MAX_RECORD_SECS  = 12

# ── Audio — TTS ───────────────────────────────────────────────────────────────
TTS_SAMPLE_RATE  = 22_050
TTS_CHANNELS     = 1
TTS_CHUNK_SIZE   = 1_024
TTS_SPEAKER      = "simran"
TTS_MODEL        = "bulbul:v3"

# ── Audio — Recording ─────────────────────────────────────────────────────────
REC_MIC_RATE    = 16_000
REC_BOT_RATE    = 22_050
REC_OUTPUT_RATE = 16_000

# ── Fan noise ─────────────────────────────────────────────────────────────────
FAN_VOLUME       = 0.4
HONK_VOLUME      = 0.5
HONK_MIN_SEC     = 45    # minimum seconds between honks (overridden in FanNoise)
HONK_MAX_SEC     = 90    # maximum seconds between honks (overridden in FanNoise)
FAN_BLOCK_SIZE   = 1_024

# ── Agent / conversation ──────────────────────────────────────────────────────
MAX_REFUSALS         = 5      # escalate after this many refusals
MAX_PTP_BUCKET       = 2      # maximum PTP attempts per customer
MAX_EMPTY_TURNS      = 3      # hang up after N consecutive empty STT results
MAX_CALL_SECONDS     = 600    # hard call timeout (10 minutes)
CONVERSATION_HISTORY_WINDOW = 20   # rolling turns kept in Gemini chat

# ── LLM ───────────────────────────────────────────────────────────────────────
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
LLM_TEMPERATURE = 0.4
LLM_RETRY_COUNT = 3
LLM_RETRY_DELAY = 1.0   # seconds, doubled each retry

# ── Sarvam STT ────────────────────────────────────────────────────────────────
SARVAM_STT_URL   = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_MODEL = "saaras:v3"
SARVAM_TIMEOUT   = 30   # seconds

# ── Sarvam TTS ────────────────────────────────────────────────────────────────
SARVAM_TTS_URL     = "https://api.sarvam.ai/text-to-speech/stream"
SARVAM_TTS_TIMEOUT = 30

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR  = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "calls.jsonl"   # one JSON object per line, append-only
LOGS_DIR = BASE_DIR / "logs"