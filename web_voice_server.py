"""
web_voice_server.py
===================
FastAPI WebSocket server for browser-based voice calls.

This replaces backend mic/speaker audio with browser mic/speaker audio:
    Browser mic -> WebSocket -> STT -> agent_core -> TTS -> WebSocket -> Browser speaker

Run:
    python -m uvicorn web_voice_server:app --host 0.0.0.0 --port 8000 --reload

Open:
    http://localhost:8000/call
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_core import RecoveryAgentSession, load_customers, make_session
from voice_io import STT_ERROR_TOKEN, rms_pcm16, sarvam_stt_from_pcm16, sarvam_tts_wav, _resolve_audio_file

try:
    from config import DATA_FILE
except Exception:
    DATA_FILE = Path(__file__).parent / "data.json"

BASE_DIR = Path(__file__).parent.resolve()
FRONTEND_DIR = BASE_DIR / "frontend"
DYNAMIC_CALLS_DIR = BASE_DIR / "dynamic_call_records"
SESSIONS: dict[str, "CallRuntime"] = {}

app = FastAPI(title="Loan Recovery Web Voice API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# API key guard for write endpoints. Optional in local dev.
# ---------------------------------------------------------------------------

def _api_key_enabled() -> bool:
    return bool(os.getenv("AGENT_API_KEY", "").strip())


def _check_key(key: Optional[str]) -> None:
    expected = os.getenv("AGENT_API_KEY", "").strip()
    if expected and key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def resolve_customer_ref(customers: list[dict[str, Any]], ref: Any = 0) -> tuple[int, dict[str, Any]]:
    """Resolve a UI customer reference.

    Supports:
      - zero-based row index: 0, 1, 2...
      - loan_id, list_id, user_id
      - phone_number / alt_phone / refrence_mobile

    This prevents users from typing loan_id=701 into an index-only field and
    getting "invalid customer index".
    """
    raw = str(ref if ref is not None else "0").strip()
    if raw == "":
        raw = "0"

    # 1) Try exact numeric row index first, but only if it is in range.
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(customers):
            return idx, customers[idx]

    # 2) Match known business identifiers and phone fields.
    candidate_fields = (
        "loan_id", "list_id", "user_id", "phone_number",
        "alt_phone", "refrence_mobile", "reference_mobile",
        "title", "first_name",
    )
    raw_norm = raw.lower()
    raw_digits = "".join(ch for ch in raw if ch.isdigit())

    for i, c in enumerate(customers):
        for field in candidate_fields:
            val = c.get(field)
            if val is None:
                continue
            val_str = str(val).strip()
            if val_str.lower() == raw_norm:
                return i, c
            val_digits = "".join(ch for ch in val_str if ch.isdigit())
            if raw_digits and val_digits and val_digits == raw_digits:
                return i, c

    raise HTTPException(
        status_code=404,
        detail=(
            f"Customer not found for '{raw}'. Use row index 0-{max(len(customers)-1, 0)}, "
            "loan_id, phone_number, list_id, or user_id."
        ),
    )



# ---------------------------------------------------------------------------
# Frontend-provided customer data
# ---------------------------------------------------------------------------

def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value).strip()))
    except Exception:
        return default


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def normalize_frontend_customer(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize customer/lead data supplied from the browser.

    The agent expects the same keys as the earlier data.json, but the frontend
    should not have to provide every optional field. This function fills safe
    defaults and validates the minimum required call fields.
    """
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="customer_data must be a JSON object")

    first_name = _clean_str(data.get("first_name") or data.get("name") or data.get("customer_name"))
    phone = _clean_str(data.get("phone_number") or data.get("phone") or data.get("mobile"))
    loan_id_raw = data.get("loan_id") or data.get("loanId") or data.get("Loan_id") or data.get("id")
    due_amount = _as_int(data.get("due_amount") or data.get("installment_amount") or data.get("emi_amount"), 0)
    loan_date = _clean_str(data.get("Loan_date") or data.get("loan_date") or data.get("date"))

    errors = []
    if not first_name:
        errors.append("first_name is required")
    if due_amount <= 0:
        errors.append("due_amount/installment_amount must be greater than 0")
    if loan_id_raw in (None, ""):
        errors.append("loan_id is required")
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    # Keep all original extra fields, but normalize the important keys.
    c = dict(data)
    c["source"] = c.get("source") or "frontend"
    c["function"] = c.get("function") or "frontend_start_call"
    c["first_name"] = first_name
    c["phone_number"] = phone
    c["loan_id"] = _as_int(loan_id_raw, 0) if str(loan_id_raw).strip().isdigit() else _clean_str(loan_id_raw)
    c["due_amount"] = due_amount
    c["installment_amount"] = _as_int(c.get("installment_amount"), due_amount) or due_amount
    c["Loan_date"] = loan_date or date.today().isoformat()
    c["dealer_name"] = _clean_str(c.get("dealer_name"), "")
    c["PTP_bucket"] = str(_as_int(c.get("PTP_bucket"), 0))
    c["Max_ptp_extension"] = _as_int(c.get("Max_ptp_extension") or c.get("max_ptp_extension"), 3)
    c["exetended_ptp_date"] = _clean_str(c.get("exetended_ptp_date") or c.get("extended_ptp_date"), "")
    c["email"] = _clean_str(c.get("email"), "")
    c["title"] = _clean_str(c.get("title"), f"FRONTEND-{c['loan_id']}")
    c["list_id"] = _as_int(c.get("list_id"), 0)
    c["user_id"] = _as_int(c.get("user_id"), 0)
    c["alt_name"] = _clean_str(c.get("alt_name"), "")
    c["alt_phone"] = _clean_str(c.get("alt_phone"), "")
    c["last_name"] = _clean_str(c.get("last_name"), "")
    c["refrence_mobile"] = _clean_str(c.get("refrence_mobile") or c.get("reference_mobile"), "")
    c["postal_code"] = _clean_str(c.get("postal_code"), "")
    return c


def persist_dynamic_call_snapshot(call_id: str, runtime: "CallRuntime") -> None:
    """Save frontend-supplied call snapshot without touching data.json."""
    try:
        DYNAMIC_CALLS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "call_id": call_id,
            "created_at": runtime.started_at,
            "saved_at": time.time(),
            "customer": runtime.session.customer,
            "state": runtime.session._state(),
            "events": runtime.session.events,
            "transcript": runtime.transcript,
        }
        (DYNAMIC_CALLS_DIR / f"{call_id}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[dynamic snapshot] failed: {e}")

# ---------------------------------------------------------------------------
# Audio VAD / utterance detector
# ---------------------------------------------------------------------------

@dataclass
class AudioDetector:
    sample_rate: int = 16000
    # Main utterance detection. Browser echoCancellation/noiseSuppression should be enabled.
    speech_rms: float = 420.0
    silence_ms_to_end: int = 700
    min_utterance_ms: int = 350
    max_utterance_ms: int = 12000
    pre_roll_ms: int = 250
    # Barge-in: stronger near-field requirement to avoid background conversations.
    barge_rms: float = 850.0
    barge_peak_rms: float = 1200.0
    barge_score_ms_to_trigger: int = 1450

    chunks: list[bytes] = field(default_factory=list)
    pre_roll: list[bytes] = field(default_factory=list)
    speech_started: bool = False
    silence_ms: int = 0
    speech_ms: int = 0
    total_ms: int = 0
    barge_score_ms: int = 0
    barge_peak: float = 0.0
    barge_triggered_current_utterance: bool = False

    def reset(self) -> None:
        self.chunks.clear()
        self.speech_started = False
        self.silence_ms = 0
        self.speech_ms = 0
        self.total_ms = 0
        self.barge_score_ms = 0
        self.barge_peak = 0.0
        self.barge_triggered_current_utterance = False

    def push(self, pcm16: bytes, *, bot_playing: bool) -> tuple[Optional[bytes], bool, dict[str, Any]]:
        """Return (completed_utterance_pcm, barge_trigger, debug)."""
        if not pcm16:
            return None, False, {}
        chunk_ms = int((len(pcm16) / 2 / self.sample_rate) * 1000)
        chunk_ms = max(1, chunk_ms)
        level = rms_pcm16(pcm16)
        is_voice = level >= self.speech_rms

        # Maintain pre-roll even before speech starts.
        self.pre_roll.append(pcm16)
        max_pre_chunks = max(1, int(self.pre_roll_ms / chunk_ms))
        if len(self.pre_roll) > max_pre_chunks:
            self.pre_roll = self.pre_roll[-max_pre_chunks:]

        if is_voice:
            if not self.speech_started:
                self.speech_started = True
                self.chunks = list(self.pre_roll)
                self.silence_ms = 0
                self.speech_ms = 0
                self.total_ms = 0
            self.chunks.append(pcm16)
            self.speech_ms += chunk_ms
            self.silence_ms = 0
        elif self.speech_started:
            self.chunks.append(pcm16)
            self.silence_ms += chunk_ms

        if self.speech_started:
            self.total_ms += chunk_ms

        # Barge-in: only trigger for close/direct speech while bot audio is currently playing.
        barge_trigger = False
        if bot_playing and self.speech_started and not self.barge_triggered_current_utterance:
            if level >= self.barge_rms:
                self.barge_score_ms += chunk_ms
                self.barge_peak = max(self.barge_peak, level)
            else:
                # decay instead of hard reset; speech has syllable gaps
                self.barge_score_ms = max(0, self.barge_score_ms - int(chunk_ms * 0.6))
            if self.barge_score_ms >= self.barge_score_ms_to_trigger and self.barge_peak >= self.barge_peak_rms:
                self.barge_triggered_current_utterance = True
                barge_trigger = True

        complete = None
        if self.speech_started:
            too_long = self.total_ms >= self.max_utterance_ms
            enough_speech = self.speech_ms >= self.min_utterance_ms
            enough_silence = self.silence_ms >= self.silence_ms_to_end
            if (enough_speech and enough_silence) or too_long:
                complete = b"".join(self.chunks)
                self.reset()

        debug = {
            "rms": int(level),
            "speech_ms": self.speech_ms,
            "silence_ms": self.silence_ms,
            "barge_score_ms": self.barge_score_ms,
            "barge_peak": int(self.barge_peak),
            "bot_playing": bot_playing,
        }
        return complete, barge_trigger, debug


@dataclass
class CallRuntime:
    call_id: str
    session: RecoveryAgentSession
    detector: AudioDetector = field(default_factory=AudioDetector)
    bot_playing: bool = False
    # True only after live barge-in has been triggered. Without this, bot echo/background
    # speech while the bot is talking can be sent to STT and make the agent repeat itself.
    barge_capture_active: bool = False
    # Prevent overlapping STT -> agent -> TTS tasks. Concurrent tasks caused repeated/out-of-order replies.
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    started_at: float = field(default_factory=time.time)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    last_debug_sent: float = 0.0
    last_playback_ended_at: float = 0.0
    last_customer_text: str = ""
    last_customer_text_at: float = 0.0
    closed: bool = False
    bg_started_at: float = field(default_factory=time.time)
    next_honk_after_sec: float = field(default_factory=lambda: random.uniform(45.0, 90.0))
    honks_played: int = 0
    max_honks: int = 2
    last_event_count: int = 0

    def should_mix_honk(self) -> bool:
        """Mirror desktop FanNoise: short honk clip only occasionally."""
        if self.honks_played >= self.max_honks:
            return False
        elapsed = time.time() - self.bg_started_at
        if elapsed >= self.next_honk_after_sec:
            self.honks_played += 1
            self.next_honk_after_sec = elapsed + random.uniform(45.0, 90.0)
            return True
        return False


async def ws_send(ws: WebSocket, runtime: CallRuntime, payload: dict[str, Any]) -> None:
    async with runtime.send_lock:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))


async def send_agent_debug(ws: WebSocket, runtime: CallRuntime, result: dict[str, Any]) -> None:
    """Send only new agent_core events to the browser log panel.

    agent_core now logs each deterministic step plus each LLM guidance/response
    event. This keeps the UI transparent without re-sending the full event list
    after every customer turn.
    """
    events = result.get("events") or []
    new_events = events[runtime.last_event_count:]
    runtime.last_event_count = len(events)
    if new_events:
        await ws_send(ws, runtime, {"type": "agent_debug", "events": new_events})


async def send_bot_reply(ws: WebSocket, runtime: CallRuntime, bot_text: str, *, final: bool = False, outcome: Optional[str] = None) -> None:
    runtime.transcript.append({"speaker": "bot", "text": bot_text, "ts": time.time()})
    await ws_send(ws, runtime, {"type": "bot_text", "text": bot_text, "final": final, "outcome": outcome})
    try:
        include_honk = runtime.should_mix_honk()
        wav_bytes = await asyncio.to_thread(
            sarvam_tts_wav,
            bot_text,
            add_background=False,  # continuous browser-side background bed is used instead
            include_honk=False,
        )
        await ws_send(ws, runtime, {
            "type": "bot_audio_wav",
            "audio": base64.b64encode(wav_bytes).decode("ascii"),
            "mime": "audio/wav",
            "final": final,
            "outcome": outcome,
        })
    except Exception as e:
        await ws_send(ws, runtime, {"type": "error", "where": "tts", "message": str(e)})


async def process_customer_audio(ws: WebSocket, runtime: CallRuntime, pcm16: bytes) -> None:
    """Run STT -> agent -> TTS for one completed customer utterance.

    This function is deliberately single-flight per call session. If multiple utterances are
    generated by echo/background noise, only one can be processed at a time; the rest are
    ignored upstream. This prevents the repeated same bot line / out-of-order reply problem.
    """
    if runtime.closed or runtime.session.closed:
        return

    if runtime.processing_lock.locked():
        await ws_send(ws, runtime, {"type": "log", "message": "Dropped audio while previous STT/agent task was still processing"})
        return

    async with runtime.processing_lock:
        if runtime.closed or runtime.session.closed:
            return
        await ws_send(ws, runtime, {"type": "stt_started"})
        text = await asyncio.to_thread(sarvam_stt_from_pcm16, pcm16, runtime.detector.sample_rate)
        if text == STT_ERROR_TOKEN:
            await ws_send(ws, runtime, {"type": "transcript", "speaker": "system", "text": "STT service error"})
            result = runtime.session.handle_user_text(STT_ERROR_TOKEN)
        elif not text.strip():
            await ws_send(ws, runtime, {"type": "transcript", "speaker": "system", "text": "No clear speech detected"})
            return
        else:
            # Drop exact duplicate transcripts produced by VAD splitting the same utterance.
            now = time.time()
            if text == runtime.last_customer_text and now - runtime.last_customer_text_at < 2.5:
                await ws_send(ws, runtime, {"type": "log", "message": f"Dropped duplicate transcript: {text}"})
                return
            runtime.last_customer_text = text
            runtime.last_customer_text_at = now

            runtime.transcript.append({"speaker": "customer", "text": text, "ts": now})
            await ws_send(ws, runtime, {"type": "transcript", "speaker": "customer", "text": text})
            result = runtime.session.handle_user_text(text)

        persist_dynamic_call_snapshot(runtime.call_id, runtime)
        await ws_send(ws, runtime, {"type": "agent_state", "state": result.get("state", {}), "customer": runtime.session.customer})
        await send_agent_debug(ws, runtime, result)
        bot_text = result.get("bot_text", "")
        if bot_text:
            await send_bot_reply(
                ws,
                runtime,
                bot_text,
                final=bool(result.get("should_end_call")),
                outcome=result.get("outcome"),
            )
        if result.get("should_end_call"):
            runtime.closed = True
            persist_dynamic_call_snapshot(runtime.call_id, runtime)
            await ws_send(ws, runtime, {"type": "call_ended", "outcome": result.get("outcome"), "events": result.get("events", []), "customer": runtime.session.customer, "state": runtime.session._state()})


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    index = FRONTEND_DIR / "call.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>Loan Recovery Web Voice API</h1><p>Put frontend/call.html beside this server or open /docs.</p>")


@app.get("/call")
async def call_page():
    index = FRONTEND_DIR / "call.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="frontend/call.html not found")
    return FileResponse(str(index))




@app.get("/bg/{filename}")
async def background_audio(filename: str):
    """Serve intentional background files to the browser for continuous call-bed audio."""
    allowed = {"fan.mp3", "honk.mp3", "honk2.mp3", "honk3.mp3"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="background file not allowed")
    path = _resolve_audio_file(filename)
    if path is None:
        raise HTTPException(status_code=404, detail=f"{filename} not found near project/config paths")
    return FileResponse(str(path), media_type="audio/mpeg")


@app.get("/health")
async def health():
    return {"ok": True, "time": time.time(), "sessions": len(SESSIONS)}


@app.get("/customer")
async def customer(index: Optional[int] = None, ref: Optional[str] = None):
    customers = load_customers(DATA_FILE)
    lookup = ref if ref is not None else (index if index is not None else 0)
    resolved_index, c = resolve_customer_ref(customers, lookup)
    return {
        "index": resolved_index,
        "customer": c,
        "ptp_available": int(c.get("PTP_bucket", 0)) < 2,
    }


class CustomerValidateRequest(BaseModel):
    customer_data: dict[str, Any]


@app.post("/customer/validate")
async def validate_customer(req: CustomerValidateRequest):
    c = normalize_frontend_customer(req.customer_data)
    return {
        "ok": True,
        "customer": c,
        "ptp_available": _as_int(c.get("PTP_bucket"), 0) < 2 and _as_int(c.get("Max_ptp_extension"), 0) > 0,
    }


class TextTestRequest(BaseModel):
    text: str
    customer_index: Optional[int] = None
    customer_ref: Optional[str] = None
    customer_data: Optional[dict[str, Any]] = None


@app.post("/test/text")
async def test_text(req: TextTestRequest):
    if req.customer_data:
        customer_record = normalize_frontend_customer(req.customer_data)
        session = RecoveryAgentSession(customer_record, all_customers=None, customer_index=0, data_file=DATA_FILE)
        opening = session.start()
        response = session.handle_user_text(req.text)
        return {"opening": opening, "response": response, "source": "frontend_customer_data", "customer": session.customer}

    customers = load_customers(DATA_FILE)
    lookup = req.customer_ref if req.customer_ref is not None else (req.customer_index if req.customer_index is not None else 0)
    resolved_index, _ = resolve_customer_ref(customers, lookup)
    session = make_session(resolved_index, DATA_FILE)
    opening = session.start()
    response = session.handle_user_text(req.text)
    return {"opening": opening, "response": response, "resolved_index": resolved_index}


@app.get("/sessions")
async def sessions():
    return {
        "sessions": [
            {"call_id": r.call_id, "age_sec": round(time.time() - r.started_at, 1), "state": r.session._state()}
            for r in SESSIONS.values()
        ]
    }


# ---------------------------------------------------------------------------
# WebSocket route
# ---------------------------------------------------------------------------

@app.websocket("/ws/call")
async def ws_call(ws: WebSocket):
    await ws.accept()
    runtime: Optional[CallRuntime] = None
    processing_tasks: set[asyncio.Task] = set()

    try:
        await ws.send_text(json.dumps({"type": "ready", "message": "send start"}, ensure_ascii=False))
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "message": "invalid json"}))
                continue

            typ = msg.get("type")

            if typ == "start":
                call_id = str(uuid.uuid4())[:8]

                # Preferred production-web path: customer/lead data comes from the frontend.
                # In this mode we do NOT write back to data.json; PTP updates live in the
                # session and are returned/snapshotted separately.
                if isinstance(msg.get("customer_data"), dict):
                    try:
                        customer_record = normalize_frontend_customer(msg["customer_data"])
                    except HTTPException as e:
                        await ws.send_text(json.dumps({"type": "error", "message": e.detail}, ensure_ascii=False))
                        continue
                    customer_index = 0
                    session = RecoveryAgentSession(customer_record, all_customers=None, customer_index=0, data_file=DATA_FILE)
                    source = "frontend_customer_data"
                else:
                    customers = load_customers(DATA_FILE)
                    lookup = msg.get("customer_ref", msg.get("customer_index", 0))
                    try:
                        customer_index, customer_record = resolve_customer_ref(customers, lookup)
                    except HTTPException as e:
                        await ws.send_text(json.dumps({"type": "error", "message": e.detail}, ensure_ascii=False))
                        continue
                    session = RecoveryAgentSession(customer_record, all_customers=customers, customer_index=customer_index, data_file=DATA_FILE)
                    source = "data_json"

                runtime = CallRuntime(call_id=call_id, session=session)
                SESSIONS[call_id] = runtime
                start_result = session.start()
                persist_dynamic_call_snapshot(call_id, runtime)
                await ws_send(ws, runtime, {
                    "type": "call_started",
                    "call_id": call_id,
                    "source": source,
                    "customer": runtime.session.customer,
                    "state": start_result["state"],
                })
                await send_agent_debug(ws, runtime, start_result)
                await send_bot_reply(ws, runtime, start_result["bot_text"])
                continue

            if runtime is None:
                await ws.send_text(json.dumps({"type": "error", "message": "call not started"}))
                continue

            if runtime.closed and typ not in ("stop", "playback_started", "playback_ended"):
                await ws_send(ws, runtime, {"type": "log", "message": "Ignoring input because call has ended"})
                continue

            if typ == "playback_started":
                runtime.bot_playing = True
                continue
            if typ == "playback_ended":
                runtime.bot_playing = False
                runtime.last_playback_ended_at = time.time()
                # If no real barge-in was confirmed, drop any half-built utterance from bot echo.
                if not runtime.barge_capture_active:
                    runtime.detector.reset()
                continue
            if typ == "stop":
                runtime.closed = True
                result = runtime.session.stop()
                persist_dynamic_call_snapshot(runtime.call_id, runtime)
                await send_agent_debug(ws, runtime, result)
                await ws_send(ws, runtime, {"type": "call_ended", "outcome": result.get("outcome"), "events": result.get("events", []), "customer": runtime.session.customer, "state": runtime.session._state()})
                await ws.close()
                break
            if typ == "text":
                text = str(msg.get("text", ""))
                runtime.transcript.append({"speaker": "customer", "text": text, "ts": time.time()})
                await ws_send(ws, runtime, {"type": "transcript", "speaker": "customer", "text": text})
                result = runtime.session.handle_user_text(text)
                await ws_send(ws, runtime, {"type": "agent_state", "state": result.get("state", {})})
                await send_agent_debug(ws, runtime, result)
                await send_bot_reply(ws, runtime, result.get("bot_text", ""), final=bool(result.get("should_end_call")), outcome=result.get("outcome"))
                if result.get("should_end_call"):
                    runtime.closed = True
                    persist_dynamic_call_snapshot(runtime.call_id, runtime)
                    await ws_send(ws, runtime, {"type": "call_ended", "outcome": result.get("outcome"), "events": result.get("events", []), "customer": runtime.session.customer, "state": runtime.session._state()})
                continue
            if typ == "audio_chunk":
                try:
                    pcm16 = base64.b64decode(msg.get("audio", ""))
                except Exception:
                    continue

                # Do not transcribe immediately after bot playback ends. Browser echo cancellation
                # often leaves a short tail that otherwise becomes a fake customer utterance.
                now = time.time()
                in_echo_tail = (not runtime.bot_playing) and (now - runtime.last_playback_ended_at < 0.65)
                if in_echo_tail and not runtime.barge_capture_active:
                    runtime.detector.reset()
                    continue

                utterance, barge, debug = runtime.detector.push(pcm16, bot_playing=runtime.bot_playing)
                now = time.time()
                if now - runtime.last_debug_sent > 2.0:
                    runtime.last_debug_sent = now
                    await ws_send(ws, runtime, {"type": "audio_debug", "debug": debug})

                if barge:
                    # This is the only case where audio captured during bot playback is trusted.
                    runtime.bot_playing = False
                    runtime.barge_capture_active = True
                    await ws_send(ws, runtime, {
                        "type": "stop_playback",
                        "reason": "barge_in",
                        "debug": debug,
                    })

                if utterance:
                    # If bot is speaking and we did not confirm barge-in, discard it.
                    # This prevents the agent from transcribing bot echo/background talk and repeating.
                    if runtime.bot_playing and not runtime.barge_capture_active:
                        runtime.detector.reset()
                        await ws_send(ws, runtime, {"type": "log", "message": "Dropped utterance during bot playback because no barge-in was confirmed"})
                        continue

                    # If the utterance began during bot playback but never reached barge threshold, ignore it.
                    if not runtime.bot_playing and not runtime.barge_capture_active and now - runtime.last_playback_ended_at < 1.0:
                        runtime.detector.reset()
                        await ws_send(ws, runtime, {"type": "log", "message": "Dropped possible playback echo after bot audio"})
                        continue

                    runtime.barge_capture_active = False
                    task = asyncio.create_task(process_customer_audio(ws, runtime, utterance))
                    processing_tasks.add(task)
                    task.add_done_callback(lambda t: processing_tasks.discard(t))
                continue

            await ws_send(ws, runtime, {"type": "error", "message": f"unknown type: {typ}"})

    except WebSocketDisconnect:
        pass
    finally:
        if runtime is not None:
            SESSIONS.pop(runtime.call_id, None)
        for task in processing_tasks:
            task.cancel()
