# Web Voice Refactor Package

This package converts the current local mic/speaker voice agent into a web-audio architecture.

## New architecture

Old local mode:

```text
Backend laptop mic -> STT -> Agent.py -> TTS -> backend laptop speaker
```

New web mode:

```text
Browser mic -> WebSocket -> backend STT -> agent_core.py -> backend TTS -> WebSocket -> browser speaker
```

The backend no longer depends on `sounddevice`, `pyaudio`, backend microphone, or backend speaker for customer calls.

## Files

```text
agent_core.py              Conversation brain only. No mic/speaker.
voice_io.py                Sarvam STT/TTS helpers for browser audio.
web_voice_server.py        FastAPI + WebSocket server.
frontend/call.html         Browser mic/speaker interface.
requirements_web_voice.txt Python dependencies.
```

## Install

Copy these files into your existing project root beside:

```text
config.py
data.json
```

Install dependencies:

```bash
pip install -r requirements_web_voice.txt
```

Set environment variables:

```bash
set SARVAM_API_KEY=your_key_here
# optional
set AGENT_API_KEY=your_admin_api_key
```

On PowerShell:

```powershell
$env:SARVAM_API_KEY="your_key_here"
```

## Run

```bash
python -m uvicorn web_voice_server:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://localhost:8000/call
```

## How to test

1. Click **Connect**.
2. Click **Start Call**.
3. Click **Enable Mic**.
4. Speak from the browser mic.
5. The bot audio will play from the browser, not the backend speaker.

## Barge-in

The browser keeps sending mic audio even while bot audio is playing. If the backend detects near-field customer speech while the bot is playing, it sends:

```json
{"type":"stop_playback","reason":"barge_in"}
```

The browser immediately stops current bot audio and continues capturing the customer utterance for STT.

## Production notes

Browser mic works only on:

```text
localhost
HTTPS domain
```

For real production, deploy behind HTTPS with Nginx/Caddy/Cloudflare.

## Important tuning values

In `web_voice_server.py`, tune these if barge-in is too sensitive or too strict:

```python
speech_rms = 420.0
barge_rms = 850.0
barge_peak_rms = 1200.0
barge_score_ms_to_trigger = 1450
```

If background speech triggers barge-in, increase `barge_rms` and `barge_peak_rms`.

If actual customer speech does not trigger, reduce them slightly.

## Current limitation

This first web package sends complete bot WAV audio to the browser per reply. It is simpler and stable. Later you can upgrade it to true chunk streaming for lower latency.
"# Loan_Recovery" 
