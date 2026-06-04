"""
voice_io.py
===========
Browser/WebSocket voice I/O helpers.

This module has no local mic/speaker dependency. It accepts PCM16 audio bytes
from the browser, writes a temporary WAV for STT, calls Sarvam STT, and returns
TTS WAV bytes for browser playback.
"""

from __future__ import annotations

import io
import os
import random
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

try:
    from config import (
        SARVAM_STT_URL,
        SARVAM_STT_MODEL,
        SARVAM_TIMEOUT,
        SARVAM_TTS_URL,
        SARVAM_TTS_TIMEOUT,
        TTS_MODEL,
        TTS_SPEAKER,
        TTS_SAMPLE_RATE,
        BASE_DIR,
        FAN_FILE,
        HONK_FILE,
        FAN_VOLUME,
        HONK_VOLUME,
    )
except Exception:
    BASE_DIR = Path(__file__).parent
    SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
    SARVAM_STT_MODEL = "saarika:v2.5"
    SARVAM_TIMEOUT = 30
    SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech/stream"
    SARVAM_TTS_TIMEOUT = 30
    TTS_MODEL = "bulbul:v3"
    TTS_SPEAKER = "simran"
    TTS_SAMPLE_RATE = 22050
    FAN_FILE = Path(BASE_DIR) / "fan.mp3"
    HONK_FILE = Path(BASE_DIR) / "honk.mp3"
    FAN_VOLUME = 0.08
    HONK_VOLUME = 0.10

STT_ERROR_TOKEN = "__STT_ERROR__"
RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}


def _candidate_audio_paths(name_or_path: str | Path) -> list[Path]:
    """Find fan/honk files reliably in web deployments.

    In browser mode the server may be started from a different folder than
    config.BASE_DIR, so we search the configured path, cwd, this module folder,
    and their parents.
    """
    p = Path(name_or_path)
    names = [p.name] if p.name else []
    roots = []
    for root in (p.parent if p.is_absolute() else None, BASE_DIR, Path.cwd(), Path(__file__).parent, Path(__file__).parent.parent):
        if root is None:
            continue
        try:
            roots.append(Path(root))
        except Exception:
            pass
    seen, out = set(), []
    if p.is_absolute():
        out.append(p)
    for root in roots:
        for nm in names:
            cand = root / nm
            key = str(cand.resolve()) if cand.exists() else str(cand)
            if key not in seen:
                seen.add(key)
                out.append(cand)
    return out


def _resolve_audio_file(name_or_path: str | Path) -> Optional[Path]:
    for cand in _candidate_audio_paths(name_or_path):
        if cand.exists():
            return cand
    return None


def pcm16_bytes_to_wav_bytes(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


def write_wav_file(path: str | Path, pcm16: bytes, sample_rate: int = 16000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return path


def rms_pcm16(pcm16: bytes) -> float:
    if len(pcm16) < 2:
        return 0.0
    arr = np.frombuffer(pcm16[: len(pcm16) - (len(pcm16) % 2)], dtype=np.int16)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))


def duration_secs_pcm16(pcm16: bytes, sample_rate: int = 16000) -> float:
    return len(pcm16) / 2 / sample_rate


def sarvam_stt_from_pcm16(pcm16: bytes, sample_rate: int = 16000, *, language_code: str = "unknown") -> str:
    """Return transcript or STT_ERROR_TOKEN on service failure."""
    if duration_secs_pcm16(pcm16, sample_rate) < 0.25:
        return ""

    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        print("[STT] SARVAM_API_KEY missing")
        return STT_ERROR_TOKEN

    wav_bytes = pcm16_bytes_to_wav_bytes(pcm16, sample_rate)
    files = {"file": ("browser_audio.wav", wav_bytes, "audio/wav")}
    headers = {"api-subscription-key": api_key}
    data = {"model": SARVAM_STT_MODEL, "language_code": language_code, "with_timestamps": "false"}

    start = time.perf_counter()
    for attempt in range(1, 4):
        try:
            resp = requests.post(
                SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=SARVAM_TIMEOUT,
            )
            if resp.status_code == 200:
                result = resp.json()
                text = (result.get("transcript") or "").strip()
                ms = int((time.perf_counter() - start) * 1000)
                print(f"[STT] {ms}ms lang={result.get('language_code', '?')} text={text!r}")
                return text

            print(f"[STT] HTTP {resp.status_code} attempt {attempt}/3: {resp.text[:200]}")
            if resp.status_code in RETRYABLE_HTTP:
                time.sleep(0.5 * attempt + random.uniform(0, 0.3))
                continue
            return STT_ERROR_TOKEN
        except requests.exceptions.Timeout as e:
            print(f"[STT] timeout attempt {attempt}/3: {e}")
            time.sleep(0.5 * attempt + random.uniform(0, 0.3))
        except Exception as e:
            print(f"[STT] error: {e}")
            return STT_ERROR_TOKEN
    return STT_ERROR_TOKEN


def detect_tts_language(text: str) -> str:
    # Script-based language detection for Sarvam TTS target language.
    if any("\u0900" <= ch <= "\u097F" for ch in text):
        return "hi-IN"
    if any("\u0A80" <= ch <= "\u0AFF" for ch in text):
        return "gu-IN"
    if any("\u0B80" <= ch <= "\u0BFF" for ch in text):
        return "ta-IN"
    if any("\u0C00" <= ch <= "\u0C7F" for ch in text):
        return "te-IN"
    if any("\u0C80" <= ch <= "\u0CFF" for ch in text):
        return "kn-IN"
    if any("\u0D00" <= ch <= "\u0D7F" for ch in text):
        return "ml-IN"
    if any("\u0980" <= ch <= "\u09FF" for ch in text):
        return "bn-IN"
    if any("\u0A00" <= ch <= "\u0A7F" for ch in text):
        return "pa-IN"
    # Hinglish markers should be spoken as Hindi.
    tl = text.lower()
    if any(w in tl for w in ["aap", "nahi", "payment", "rupaye", "kar", "ji", "ptp"]):
        return "hi-IN"
    return "en-IN"



# ---------------------------------------------------------------------------
# Intentional background audio for web mode
# ---------------------------------------------------------------------------

def _linear_gain_to_db(value: float, *, default_db: float) -> float:
    """Convert old config linear volume values to dB safely."""
    try:
        value = float(value)
    except Exception:
        return default_db
    if value <= 0:
        return -120.0
    # Direct browser playback is cleaner than speaker-recapture, but if the
    # gain is too low users think the fan is missing. Keep it audible but below speech.
    return max(-32.0, min(-10.0, 20.0 * np.log10(value) - 3.0))


def _load_audio_segment(path: str | Path):
    try:
        from pydub import AudioSegment
    except Exception as e:
        print(f"[BG] pydub unavailable; background disabled: {e}")
        return None

    try:
        resolved = _resolve_audio_file(path)
        if resolved is None:
            print(f"[BG] file not found for {path}; checked: {[str(x) for x in _candidate_audio_paths(path)]}")
            return None
        print(f"[BG] loading background file: {resolved}")
        return AudioSegment.from_file(str(resolved)).set_channels(1)
    except Exception as e:
        print(f"[BG] could not load {path}: {e}")
        return None


def _loop_segment(seg, duration_ms: int):
    if seg is None or duration_ms <= 0:
        return None
    if len(seg) == 0:
        return None
    reps = int(duration_ms / len(seg)) + 2
    return (seg * reps)[:duration_ms]


def mix_background_into_wav(wav_bytes: bytes, *, include_honk: bool = False) -> bytes:
    """Mix fan bed and optional short honk into Sarvam WAV bytes.

    This replaces the old desktop FanNoise OutputStream for web production.
    The customer hears the mixed WAV in the browser, so no backend speaker or
    local microphone recapture is required.
    """
    try:
        from pydub import AudioSegment
    except Exception as e:
        print(f"[BG] pydub unavailable; returning clean TTS: {e}")
        return wav_bytes

    try:
        bot = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav").set_channels(1)
    except Exception as e:
        print(f"[BG] cannot parse TTS WAV; returning clean TTS: {e}")
        return wav_bytes

    mixed = bot

    fan = _load_audio_segment(FAN_FILE)
    if fan is not None:
        fan = fan.set_frame_rate(bot.frame_rate).set_sample_width(bot.sample_width)
        fan_loop = _loop_segment(fan, len(bot))
        if fan_loop is not None:
            fan_gain = _linear_gain_to_db(FAN_VOLUME, default_db=-28.0)
            mixed = mixed.overlay(fan_loop.apply_gain(fan_gain))
            print(f"[BG] fan mixed into bot audio gain={fan_gain:.1f}dB")

    if include_honk:
        raw_candidates = [
            Path(BASE_DIR) / "honk.mp3",
            Path(BASE_DIR) / "honk2.mp3",
            Path(BASE_DIR) / "honk3.mp3",
            Path(HONK_FILE),
        ]
        candidates = []
        seen = set()
        for raw in raw_candidates:
            resolved = _resolve_audio_file(raw)
            if resolved is not None and str(resolved) not in seen:
                seen.add(str(resolved))
                candidates.append(resolved)
        if candidates:
            honk = _load_audio_segment(random.choice(candidates))
            if honk is not None:
                honk = honk.set_frame_rate(bot.frame_rate).set_sample_width(bot.sample_width)
                clip_ms = min(len(honk), random.randint(1500, 3500))
                if clip_ms > 250:
                    start_ms = random.randint(0, max(0, len(honk) - clip_ms))
                    clip = honk[start_ms:start_ms + clip_ms].fade_out(min(400, clip_ms // 3))
                    honk_gain = _linear_gain_to_db(HONK_VOLUME, default_db=-24.0)
                    # Don't put honk exactly at time 0; it sounds fake and can mask the first word.
                    pos_ms = random.randint(250, max(250, max(250, len(bot) - clip_ms)))
                    mixed = mixed.overlay(clip.apply_gain(honk_gain), position=pos_ms)
                    print(f"[BG] honk mixed into bot audio gain={honk_gain:.1f}dB pos={pos_ms}ms")

    out = io.BytesIO()
    mixed.export(out, format="wav")
    return out.getvalue()


def sarvam_tts_wav(text: str, *, language_code: Optional[str] = None, add_background: bool = False, include_honk: bool = False) -> bytes:
    """Return WAV bytes for browser playback. Raises RuntimeError on failure."""
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY missing")

    lang = language_code or detect_tts_language(text)
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "target_language_code": lang,
        "speaker": TTS_SPEAKER,
        "model": TTS_MODEL,
        "speech_sample_rate": TTS_SAMPLE_RATE,
        "output_audio_codec": "wav",
        "enable_preprocessing": True,
    }
    start = time.perf_counter()
    with requests.post(SARVAM_TTS_URL, headers=headers, json=payload, stream=True, timeout=SARVAM_TTS_TIMEOUT) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"TTS API error {resp.status_code}: {resp.text[:300]}")
        chunks = [c for c in resp.iter_content(chunk_size=4096) if c]
    audio = b"".join(chunks)
    print(f"[TTS] {int((time.perf_counter() - start)*1000)}ms lang={lang} bytes={len(audio)}")
    if add_background:
        audio = mix_background_into_wav(audio, include_honk=include_honk)
    return audio
