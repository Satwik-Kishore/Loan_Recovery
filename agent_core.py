"""
agent_core.py
=============
Web-safe conversation brain for the EMI recovery voice agent.

This file intentionally contains no microphone, speaker, PyAudio or sounddevice
code. The browser/telephony layer should do audio I/O and call this class with
text transcripts.

It ports the production Agent.py call-flow into a deterministic web session:
- identity confirmation
- due EMI prompt
- PTP negotiation with requested-day support
- strict no-PTP guard
- deterministic PTP creation + final call close
- customer question routing before refusal/reason logic
- reason gate so statements/questions are not treated as reasons
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from config import (
        DATA_FILE,
        MAX_PTP_BUCKET,
        MAX_REFUSALS,
        GEMINI_MODEL,
        LLM_TEMPERATURE,
        LLM_RETRY_COUNT,
        LLM_RETRY_DELAY,
    )
except Exception:
    DATA_FILE = Path(__file__).parent / "data.json"
    MAX_PTP_BUCKET = 2
    MAX_REFUSALS = 5
    GEMINI_MODEL = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.65
    LLM_RETRY_COUNT = 2
    LLM_RETRY_DELAY = 0.4

STT_ERROR_TOKEN = "__STT_ERROR__"


class Outcome(str, Enum):
    COMMITTED = "COMMITTED"
    PTP_CREATED = "PTP_CREATED"
    ESCALATED = "ESCALATED"
    WRONG_PERSON = "WRONG_PERSON"
    CUSTOMER_ENDED = "CUSTOMER_ENDED"
    CALL_ABANDONED = "CALL_ABANDONED"


class Stage(str, Enum):
    IDENTITY = "IDENTITY"
    ASK_PAYMENT = "ASK_PAYMENT"
    NEGOTIATION = "NEGOTIATION"
    NO_PTP = "NO_PTP"
    REASON = "REASON"
    CLOSED = "CLOSED"


@dataclass
class AgentResult:
    bot_text: str
    should_end_call: bool = False
    outcome: Optional[str] = None
    events: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bot_text": self.bot_text,
            "should_end_call": self.should_end_call,
            "outcome": self.outcome,
            "events": self.events,
            "state": self.state,
        }


# ---------------------------------------------------------------------------
# Text normalization and intent helpers
# ---------------------------------------------------------------------------

_INDIC_DIGIT_TRANSLATION = str.maketrans({
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
    "૦": "0", "૧": "1", "૨": "2", "૩": "3", "૪": "4",
    "૫": "5", "૬": "6", "૭": "7", "૮": "8", "૯": "9",
})


def _normalize_deva(text: str) -> str:
    return text.replace("\u0901", "\u0902")


def _norm(text: str) -> str:
    text = _normalize_deva(text or "").translate(_INDIC_DIGIT_TRANSLATION)
    text = text.strip().lower()
    text = re.sub(r"[।॥,!?;:\-_/\\()\[\]{}\"']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text or "")


def _has_gujarati(text: str) -> bool:
    return any("\u0A80" <= ch <= "\u0AFF" for ch in text or "")


def _detect_lang(text: str, default: str = "hi") -> str:
    if _has_gujarati(text):
        return "gu"
    if _has_devanagari(text):
        return "hi"
    t = (text or "").lower()
    if any(w in t for w in ("kem", "shu", "tame", "chho", "mane", "paisaa", "divas")):
        return "gu"
    if any(w in t for w in ("hai", "nahi", "mujhe", "aap", "kar", "paise", "din")):
        return "hi"
    return default or "hi"


def _gu_name(name: str) -> str:
    mapping = {"rahul": "રાહુલ", "patel": "પટેલ", "amit": "અમિત", "shah": "શાહ", "ramesh": "રમેશ"}
    return " ".join(mapping.get(part.lower(), part) for part in str(name or "").split()).strip() or str(name or "customer")


def _hi_name(name: str) -> str:
    """Tiny name renderer so common demo names do not sound half-English."""
    parts = []
    mapping = {"rahul": "राहुल", "patel": "पटेल", "amit": "अमित", "shah": "शाह", "ramesh": "रमेश"}
    for part in str(name or "").split():
        parts.append(mapping.get(part.lower(), part))
    return " ".join(parts).strip() or str(name or "customer")


def _contains_any(text: str, words: tuple[str, ...] | list[str]) -> bool:
    t = _norm(text)
    return any(w.lower() in t for w in words)


def _contains_two_or_three_without_unit(text: str) -> bool:
    """Detect unclear day answers like 'do teen mujhe' after a PTP question."""
    t = _norm(text)
    has_two = any(x in t.split() for x in ("दो", "2", "do", "two", "બે"))
    has_three = any(x in t.split() for x in ("तीन", "3", "teen", "three", "ત્રણ"))
    return has_two and has_three and extract_requested_days(text) is None


def customer_wants_to_end(text: str) -> bool:
    return _contains_any(text, ("bye", "बाय", "phone rakho", "फोन रख", "band karo", "बंद करो", "कॉल काट", "call cut"))


def looks_like_background_or_offtopic(text: str) -> bool:
    """Drop obvious unrelated browser/background transcripts only."""
    t = _norm(text)
    if not t or _has_devanagari(text) or _has_gujarati(text):
        return False
    useful = (
        "pay", "payment", "emi", "loan", "ptp", "extension", "time", "day", "days", "fine", "late",
        "fee", "amount", "cibil", "confirm", "yes", "no", "sorry", "cannot", "can't", "money", "salary", "today",
    )
    if any(w in t for w in useful):
        return False
    unrelated = ("new orleans", "history", "agriculture", "variable", "beginning", "next one", "inhibit")
    return any(w in t for w in unrelated) or (len(t.split()) >= 4 and not any(w in t for w in useful))


_DAY_WORDS = {
    "एक": 1, "1": 1, "ek": 1, "one": 1,
    "दो": 2, "2": 2, "do": 2, "two": 2,
    "तीन": 3, "तिन": 3, "3": 3, "teen": 3, "tin": 3, "three": 3,
    "चार": 4, "4": 4, "char": 4, "chaar": 4, "four": 4,
    "पांच": 5, "पाँच": 5, "5": 5, "panch": 5, "paanch": 5, "five": 5,
    "छह": 6, "छः": 6, "6": 6, "six": 6,
    "सात": 7, "7": 7, "seven": 7,
    # Gujarati
    "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4, "પાંચ": 5, "છ": 6, "છે": 6, "સાત": 7,
}
_DAY_UNIT_PATTERN = r"(?:din|day|days|दीन|दिन|दिवस|diwas|divas|દિવસ|દિન|divas|hafte|week|weeks|हफ्ते|हफ़्ते)"


def extract_requested_days(text: str) -> Optional[int]:
    raw = _norm(text)
    if not raw:
        return None

    def is_negated(end: int) -> bool:
        after = raw[end : end + 16]
        return any(x in after for x in ("नहीं", "नही", "nahi", "nahin", "not", "no "))

    candidates: list[tuple[int, int]] = []
    for m in re.finditer(rf"\b([0-9]{{1,2}})\s*{_DAY_UNIT_PATTERN}\b", raw, flags=re.IGNORECASE):
        if not is_negated(m.end()):
            candidates.append((m.start(), int(m.group(1))))

    for word, value in _DAY_WORDS.items():
        w = re.escape(word.lower())
        for m in re.finditer(rf"(?:^|\s){w}\s*{_DAY_UNIT_PATTERN}\b", raw, flags=re.IGNORECASE):
            if not is_negated(m.end()):
                candidates.append((m.start(), value))
        for m in re.finditer(rf"{_DAY_UNIT_PATTERN}\s*(?:ka|का|की|के)?\s*{w}(?:\s|$)", raw, flags=re.IGNORECASE):
            if not is_negated(m.end()):
                candidates.append((m.start(), value))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def is_identity_confirmed(text: str) -> bool:
    t = _norm(text)
    wrong = ("wrong", "गलत", "not rahul", "राहुल नहीं", "nahi rahul", "नहीं राहुल")
    if any(w in t for w in wrong):
        return False
    return any(w in t for w in ("हाँ", "हां", "haan", "ha", "yes", "मैं", "main", "बोल रहा", "बोल रही", "speaking", "હા", "હું", "બોલું"))


def is_wrong_person(text: str) -> bool:
    return _contains_any(text, ("wrong number", "गलत नंबर", "wrong person", "मैं राहुल नहीं", "main rahul nahi", "not rahul", "ખોટો નંબર", "હું રાહુલ નથી"))


def asks_fine_or_late_fee(text: str) -> bool:
    return _contains_any(text, (
        "fine", "late fee", "late fees", "penalty", "charge", "charges", "extra charge",
        "फाइन", "फाईन", "लेट फीस", "लेट फी", "लेटफीस", "जुर्माना", "पेनल्टी", "चार्ज", "कितना लगेगा", "कितने का", "फाइन चढ़ेगा", "गिलेट फेस", "લેટ ફી", "ફાઇન", "પેનલ્ટી", "ચાર્જ", "કેટલું લાગશે",
    ))


def asks_loan_details(text: str) -> bool:
    if asks_fine_or_late_fee(text):
        return False
    return _contains_any(text, (
        "which emi", "what emi", "कौन सी", "कौनसा", "कौन सा", "details", "detail", "डिटेल", "emi kaunsi", "ईएमआई", "loan id", "लोन", "વિગત", "ડિટેલ", "કઈ emi", "લોન", "ઇએમઆઈ",
    ))


def asks_amount(text: str) -> bool:
    return _contains_any(text, ("how much amount", "amount", "कितना amount", "कितने रुपये", "कितना पैसा", "कितना भुगतान", "राशि", "अमाउंट", "કેટલા રૂપિયા", "કેટલું amount", "રકમ"))


def asks_cibil(text: str) -> bool:
    return _contains_any(text, ("cibil", "सिबिल", "credit score", "क्रेडिट स्कोर"))


def asks_answer_first(text: str) -> bool:
    return _contains_any(text, (
        "answer my question", "first answer", "answer first", "pehle jawab", "pehle bata", "पहले आप मेरी बात", "पहले मेरा सवाल", "पहले जवाब", "मेरी बात का जवाब", "सवाल का जवाब", "पहले बताइए", "पहले बताओ", "उसके बाद", "પહેલા જવાબ", "પહેલા કહો", "મારા સવાલનો જવાબ", "પછી",
    ))


def mentions_ptp_or_extension(text: str) -> bool:
    t = _norm(text)
    if extract_requested_days(text) is not None:
        return True
    return any(w in t for w in (
        "ptp", "पीटीपी", "peti", "पेटी", "पिटीपी", "promise to pay", "extension", "extention", "एक्सटेंशन", "time", "समय", "टाइम", "मोहलत", "दिन", "days", "din", "बाद में", "थोड़ा समय", "थोड़े", "थोड़ा", "પીટીપી", "પીટિપી", "ટાઈમ", "ટાઇમ", "સમય", "જોઈએ", "દિવસ", "મુદત", "મોહલત",
    ))


def is_explicit_ptp_confirmation(text: str) -> bool:
    t = _norm(text)
    return any(w in t for w in (
        "ठीक है", "थीक है", "ठिक है", "हाँ", "हां", "जी हाँ", "जी हां", "दे दीजिए", "दे दीजिये", "कर दीजिए", "कर दीजिये", "नोट कर", "लिख लो", "कन्फर्म", "confirm", "confirmed", "ok", "okay", "theek hai", "thik hai", "de dijiye", "kar dijiye", "note kar", "haan", "yes",
    ))


def is_payment_commitment(text: str) -> bool:
    t = _norm(text)
    if mentions_ptp_or_extension(text):
        return False
    if any(w in t for w in ("नहीं", "nahi", "not", "can't", "cannot", "नहीं कर पाऊ", "नहीं होगा", "मुमकिन नहीं")):
        return False
    return any(w in t for w in (
        "कर दूँगा", "कर दूंगा", "कर दूंगी", "भर दूँगा", "भर दूंगा", "दे दूँगा", "दे दूंगा", "pay kar", "payment kar", "i will pay", "will pay", "आज कर", "aaj kar", "हाँ मैं कर", "haan main kar", "कर दूं", "कर दूँ", "કરી દઈશ", "કરીશ", "ભરી દઈશ", "ચૂકવી દઈશ", "હું કરી દઈશ",
    ))


def refuses_payment(text: str) -> bool:
    t = _norm(text)
    if mentions_ptp_or_extension(text):
        return False
    direct_markers = (
        "नहीं कर पाऊंगा", "नहीं कर पाऊँगा", "नहीं कर पाऊंगी", "नहीं कर पाऊँगी",
        "नहीं कर पाऊ", "नही कर पाऊ", "नहीं होगा", "आज नहीं", "पेमेंट नहीं",
        "payment नहीं", "भुगतान नहीं", "क्लियर नहीं", "क्लियर नही", "clear nahi",
        "clear नहीं", "clear नही", "क्लियर नहीं कर", "लियर नहीं", "लियर नही",
        "abhi nahi", "nahi kar paunga", "not possible", "can't pay", "cannot pay",
        "पैसे नहीं", "money nahi", "paise nahi", "मेरे पास पैसे नहीं",
        "નથી કરી શકતો", "નથી કરી શકતી", "આજે નહીં", "પેમેન્ટ નહીં", "પૈસા નથી", "નથી થઈ શકે",
    )
    if any(m in t for m in direct_markers):
        return True
    # STT variants such as "मैं क्लियर नहीं कर पाऊं" or "मैं आज clear नहीं कर पाऊंगा"
    if ("नहीं" in t or "नही" in t or "nahi" in t) and any(v in t for v in ("payment", "पेमेंट", "भुगतान", "clear", "क्लियर", "कर पाऊ", "कर पाउ", "pay")):
        return True
    return False


def is_clear_reason(text: str) -> bool:
    """True only when the customer actually gave a non-payment reason.

    Ported from the production Agent.py behavior: customer questions or
    "answer me first" statements are not treated as reasons, but genuine
    hardship phrases such as medical emergency, hospital, job loss, salary
    delay, cash shortage, business loss, etc. are accepted immediately.
    """
    if asks_answer_first(text) or asks_fine_or_late_fee(text) or asks_loan_details(text) or asks_amount(text):
        return False
    return _contains_any(text, (
        # cash / income issue
        "पैसे नहीं", "पैसा नहीं", "पैसे नही", "अभी पैसे", "paise nahi", "money nahi",
        "salary", "सैलरी", "वेतन", "income", "इनकम", "कमाई", "काम नहीं",
        "नौकरी", "job", "job gaya", "नौकरी चली", "laid off",
        # hardship / emergency
        "medical", "मेडिकल", "emergency", "इमरजेंसी", "एमरजेंसी", "ઇમરજન્સી", "મેડિકલ", "હોસ્પિટલ",
        "hospital", "अस्पताल", "बीमार", "bimar", "bemar",
        "doctor", "डॉक्टर", "इलाज", "दवाई", "operation", "ऑपरेशन", "surgery",
        # business / family / other problem
        "business", "व्यापार", "loss", "नुकसान", "घर में", "family",
        "दिक्कत", "problem", "issue", "मजबूरी", "मजबूर", "દિક્કત", "સમસ્યા", "મજબૂરી", "પૈસા નથી", "મેડિકલ", "ઇમરજન્સી", "હોસ્પિટલ", "બીમાર",
    ))

def is_critical_reason(text: str) -> bool:
    """Serious hardships that should be escalated to a senior.

    This mirrors the production Agent.py escalation rule for medical emergency,
    job loss, accident, death in family, serious illness, natural disaster, etc.
    """
    return _contains_any(text, (
        # Medical / emergency
        "medical emergency", "मेडिकल इमरजेंसी", "मेडिकल एमरजेंसी",
        "hospital", "अस्पताल", "icu", "आईसीयू", "surgery", "operation", "ऑपरेशन",
        "accident", "दुर्घटना", "बीमार", "बिमार", "doctor", "डॉक्टर", "इलाज",
        "emergency", "इमरजेंसी", "एमरजेंसी", "ઇમરજન્સી", "મેડિકલ", "હોસ્પિટલ",
        # death/family emergency
        "death", "मृत्यु", "maut", "मौत", "funeral",
        # income disaster
        "job loss", "नौकरी चली", "naukri gayi", "laid off", "retrenchment",
        "business बंद", "business band",
        # force majeure
        "flood", "baadh", "बाढ़", "earthquake", "भूकंप", "fire", "आग",
    ))


class RecoveryAgentSession:
    """Deterministic policy router + fully dynamic LLM speaker.

    The Python code below decides *state and actions* only: identity verified,
    payment committed, PTP requested, PTP created, escalation, etc. Every
    customer-facing message is generated by the LLM from guidance/facts at the
    moment it is needed. This keeps the agent natural without letting the LLM
    mutate business rules or silently create PTP records.
    """

    def __init__(
        self,
        customer: dict[str, Any],
        *,
        all_customers: Optional[list[dict[str, Any]]] = None,
        customer_index: int = 0,
        data_file: Path | str = DATA_FILE,
        max_ptp_bucket: int = MAX_PTP_BUCKET,
        max_refusals: int = MAX_REFUSALS,
    ):
        self.customer = customer
        self.all_customers = all_customers
        self.customer_index = customer_index
        self.data_file = Path(data_file)
        self.max_ptp_bucket = int(max_ptp_bucket)
        self.max_refusals = int(max_refusals)

        self.stage = Stage.IDENTITY
        self.turns = 0
        self.refusals = 0
        self.push_count = 0
        self.reason_asked = False
        self.pending_ptp_days: Optional[int] = None
        self.last_customer_question: Optional[str] = None
        self.last_bot_text = ""
        self.events: list[dict[str, Any]] = []
        self.dialogue: list[dict[str, str]] = []
        self.closed = False
        self.lang = "hi"
        self.response_count_by_intent: dict[str, int] = {}

        self.debug_logs = os.getenv("AGENT_DEBUG_LOGS", "1") != "0"
        self.strict_dynamic_llm = os.getenv("AGENT_STRICT_DYNAMIC_LLM", "1") != "0"
        self.llm_model_name = os.getenv("GEMINI_MODEL", str(GEMINI_MODEL))
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", str(LLM_TEMPERATURE)))
        self.llm_retry_count = int(os.getenv("LLM_RETRY_COUNT", str(LLM_RETRY_COUNT)))
        self.llm_retry_delay = float(os.getenv("LLM_RETRY_DELAY", str(LLM_RETRY_DELAY)))
        self.llm_dynamic = bool(os.getenv("GEMINI_API_KEY")) and genai is not None and os.getenv("AGENT_USE_DYNAMIC_LLM", "1") != "0"
        self._llm_model = None
        if self.llm_dynamic:
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self._llm_model = genai.GenerativeModel(
                    model_name=self.llm_model_name,
                    system_instruction=self._system_instruction(),
                )
                self._log("llm_ready", model=self.llm_model_name, strict_dynamic=self.strict_dynamic_llm)
            except Exception as e:
                self.llm_dynamic = False
                self._llm_model = None
                self._log("llm_init_failed", error=str(e), model=self.llm_model_name)
        else:
            self._log(
                "llm_disabled",
                reason="GEMINI_API_KEY missing, google.generativeai unavailable, or AGENT_USE_DYNAMIC_LLM=0",
                strict_dynamic=self.strict_dynamic_llm,
            )

    # ------------------------------------------------------------------
    # Customer facts
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return str(self.customer.get("first_name") or "customer")

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else "customer"

    @property
    def display_name(self) -> str:
        return _gu_name(self.name) if self.lang == "gu" else _hi_name(self.name)

    @property
    def display_first(self) -> str:
        return _gu_name(self.first_name) if self.lang == "gu" else _hi_name(self.first_name)

    @property
    def due_amount(self) -> int:
        return int(float(self.customer.get("due_amount") or self.customer.get("installment_amount") or 0))

    @property
    def loan_id(self) -> Any:
        return self.customer.get("loan_id", "")

    @property
    def loan_date(self) -> str:
        return str(self.customer.get("Loan_date") or "")

    @property
    def dealer_name(self) -> str:
        return str(self.customer.get("dealer_name") or "")

    @property
    def ptp_bucket(self) -> int:
        try:
            return int(self.customer.get("PTP_bucket", 0))
        except Exception:
            return 0

    @property
    def max_ptp_days(self) -> int:
        try:
            return int(self.customer.get("Max_ptp_extension", 0))
        except Exception:
            return 0

    def ptp_available(self) -> bool:
        return self.ptp_bucket < self.max_ptp_bucket and self.max_ptp_days > 0

    def _lang_label(self) -> str:
        return {"hi": "Hindi in Devanagari script", "gu": "Gujarati script", "en": "Indian English"}.get(self.lang, "Hindi in Devanagari script")

    def _bump_intent(self, intent: str) -> int:
        self.response_count_by_intent[intent] = self.response_count_by_intent.get(intent, 0) + 1
        return self.response_count_by_intent[intent]

    def _safe_customer_facts(self) -> dict[str, Any]:
        return {
            "customer_name": self.name,
            "display_name_for_language": self.display_name,
            "due_amount_rs": self.due_amount,
            "loan_id": self.loan_id,
            "loan_date": self.loan_date,
            "dealer_name": self.dealer_name,
            "ptp_bucket_used": self.ptp_bucket,
            "max_ptp_bucket": self.max_ptp_bucket,
            "max_ptp_extension_days": self.max_ptp_days,
            "ptp_available": self.ptp_available(),
            "pending_ptp_days": self.pending_ptp_days,
        }

    def _state(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "turns": self.turns,
            "refusals": self.refusals,
            "push_count": self.push_count,
            "ptp_bucket": self.ptp_bucket,
            "max_ptp_bucket": self.max_ptp_bucket,
            "ptp_available": self.ptp_available(),
            "pending_ptp_days": self.pending_ptp_days,
            "llm_dynamic": self.llm_dynamic,
            "llm_model": self.llm_model_name,
            "strict_dynamic_llm": self.strict_dynamic_llm,
            "closed": self.closed,
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, event: str, **payload: Any) -> None:
        record = {"event": event, "ts": datetime.now().isoformat(), **payload}
        self.events.append(record)
        if self.debug_logs:
            try:
                printable = json.dumps(record, ensure_ascii=False, default=str)
            except Exception:
                printable = str(record)
            print(f"[AGENT_STEP] {printable}")

    def _visible_events_since(self, offset: int) -> list[dict[str, Any]]:
        return self.events[offset:]

    # ------------------------------------------------------------------
    # Persistence / actions
    # ------------------------------------------------------------------
    def _save_customer(self) -> None:
        if self.all_customers is None:
            return
        self.all_customers[self.customer_index] = self.customer
        tmp = self.data_file.with_suffix(self.data_file.suffix + ".tmp")
        tmp.write_text(json.dumps(self.all_customers, indent=4, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.data_file)
        self._log("customer_saved", data_file=str(self.data_file), customer_index=self.customer_index)

    def _create_ptp(self, days: int) -> dict[str, Any]:
        self._log("tool_create_ptp_requested", days=days, ptp_available=self.ptp_available(), max_ptp_days=self.max_ptp_days, ptp_bucket=self.ptp_bucket)
        if not self.ptp_available():
            result = {"status": "failed", "message": "PTP exhausted"}
            self._log("tool_create_ptp_result", result=result)
            return result
        if days < 1 or days > self.max_ptp_days:
            result = {"status": "failed", "message": f"Maximum allowed PTP is {self.max_ptp_days} days"}
            self._log("tool_create_ptp_result", result=result)
            return result
        extended_date = date.today() + timedelta(days=days)
        self.customer["PTP_bucket"] = str(self.ptp_bucket + 1)
        self.customer["exetended_ptp_date"] = str(extended_date)
        self._save_customer()
        result = {"status": "success", "extended_date": str(extended_date), "days": days}
        self._log("tool_create_ptp_result", result=result)
        return result

    # ------------------------------------------------------------------
    # Dynamic LLM speaker
    # ------------------------------------------------------------------
    def _system_instruction(self) -> str:
        return """
You are Priya, a professional female Indian EMI recovery voice agent for XYZ Finance.
The application code decides the call stage, policy checks, PTP creation, escalation, and whether the call ends.
Your job is only to write the next customer-facing sentence from the guidance.

Hard rules:
- Output only the exact words Priya should say to the customer.
- Do not output analysis, JSON, bullets, labels, tool names, markdown, or internal reasoning.
- Do not invent policy, dates, charges, loan facts, names, amounts, or PTP approval.
- Use only facts supplied in the prompt.
- Be firm but respectful. Never threaten, shame, harass, or use abusive pressure.
- If the guidance says a PTP is unavailable or exhausted, do not offer any extension.
- If the guidance says ask PTP duration after refusal, do not ask again whether they can pay today.
- Keep it concise for low-latency phone speech.
""".strip()

    def _recent_dialogue_text(self) -> str:
        if not self.dialogue:
            return "No previous customer/bot turns in this session."
        rows = []
        for item in self.dialogue[-8:]:
            speaker = item.get("speaker", "?")
            text = item.get("text", "")
            rows.append(f"{speaker}: {text}")
        return "\n".join(rows)

    def _clean_llm_text(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip().strip('"').strip("'")
        text = re.sub(r"\s+", " ", text).strip()
        # Strip common accidental labels.
        text = re.sub(r"^(Priya|Bot|Agent)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
        return text

    def _technical_llm_failure_message(self, reason: str) -> str:
        self._log("strict_dynamic_generation_failed", reason=reason)
        # This is intentionally not a business-flow template. It is only an operational fallback.
        if self.lang == "gu":
            return "માફ કરશો, હાલમાં system response generate કરી શકતું નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયત્ન કરશો."
        if self.lang == "en":
            return "Sorry, the system cannot generate the response right now. Please try again shortly."
        return "माफ़ कीजिए, अभी system response generate नहीं कर पा रहा है। कृपया थोड़ी देर बाद फिर कोशिश कीजिए।"

    def _build_dynamic_prompt(
        self,
        *,
        intent: str,
        user_text: str,
        guidance: str,
        required_facts: str = "",
        forbid: str = "",
        max_sentences: int = 2,
    ) -> str:
        facts = self._safe_customer_facts()
        return f"""
Write Priya's next spoken reply dynamically. Do not reuse a fixed template.

Language to use: {self._lang_label()}
Intent label for variation only: {intent}
Response variant number for this intent: {self.response_count_by_intent.get(intent, 0) + 1}
Current deterministic stage: {self.stage.value}
Customer's latest words: {user_text!r}
Recent dialogue:
{self._recent_dialogue_text()}

Customer/account facts allowed:
{json.dumps(facts, ensure_ascii=False, default=str)}

Policy/business guidance decided by Python:
{guidance}

Facts that must be included when relevant:
{required_facts or 'None'}

Things to avoid:
{forbid or 'None'}

Voice style:
- Natural Indian phone-call wording, not robotic.
- Acknowledge the customer's latest point briefly before moving forward.
- Maximum {max_sentences} sentences.
- No markdown. No labels. No explanation of what you are doing.
- Output only the final customer-facing reply.
""".strip()

    def _generate_reply(
        self,
        *,
        intent: str,
        user_text: str = "",
        guidance: str,
        required_facts: str = "",
        forbid: str = "",
        max_sentences: int = 2,
    ) -> str:
        variant = self._bump_intent(intent)
        prompt = self._build_dynamic_prompt(
            intent=intent,
            user_text=user_text,
            guidance=guidance,
            required_facts=required_facts,
            forbid=forbid,
            max_sentences=max_sentences,
        )
        self._log(
            "llm_guidance_prepared",
            intent=intent,
            variant=variant,
            stage=self.stage.value,
            user_text=user_text[:300],
            guidance=guidance,
            required_facts=required_facts,
            forbid=forbid,
            max_sentences=max_sentences,
            state=self._state(),
        )

        if not self.llm_dynamic or self._llm_model is None:
            return self._technical_llm_failure_message("LLM is not configured. Set GEMINI_API_KEY and keep AGENT_USE_DYNAMIC_LLM=1.")

        last_error = None
        for attempt in range(1, max(1, self.llm_retry_count) + 1):
            started = time.perf_counter()
            try:
                self._log("llm_call_started", intent=intent, attempt=attempt, model=self.llm_model_name)
                resp = self._llm_model.generate_content(
                    prompt,
                    generation_config={"temperature": self.llm_temperature, "top_p": 0.9},
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                text = self._clean_llm_text(getattr(resp, "text", "") or "")
                blocked = False
                if not text or len(text) < 4:
                    blocked = True
                    last_error = "empty LLM response"
                # Extra guardrails for known bad loops.
                low = text.lower()
                if intent in {"ask_ptp_days_after_refusal", "ptp_duration_unclear"}:
                    bad = (
                        "pay today", "payment today", "आज payment", "आज पेमेंट",
                        "clear today", "payment कर पाए", "payment करेंगे", "આજે payment",
                    )
                    if any(b in low or b in text for b in bad):
                        blocked = True
                        last_error = "guardrail blocked payment-today question during PTP duration flow"
                if "```" in text or "{\"" in text or text.lower().startswith(("i should", "i need", "the customer")):
                    blocked = True
                    last_error = "guardrail blocked non-spoken/internal output"

                self._log(
                    "llm_call_finished",
                    intent=intent,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    blocked=blocked,
                    text=text[:500],
                    error=last_error,
                )
                if not blocked:
                    return text

                prompt += "\n\nPrevious output was rejected by guardrails. Rewrite as a valid spoken reply only, obeying all restrictions."
            except Exception as e:
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_error = str(e)
                self._log("llm_call_error", intent=intent, attempt=attempt, latency_ms=latency_ms, error=last_error)
                if attempt < self.llm_retry_count:
                    time.sleep(self.llm_retry_delay * attempt)

        return self._technical_llm_failure_message(last_error or "LLM response failed validation")

    def _reply_result(
        self,
        *,
        intent: str,
        user_text: str = "",
        guidance: str,
        required_facts: str = "",
        forbid: str = "",
        max_sentences: int = 2,
        should_end: bool = False,
        outcome: Optional[Outcome | str] = None,
    ) -> dict[str, Any]:
        text = self._generate_reply(
            intent=intent,
            user_text=user_text,
            guidance=guidance,
            required_facts=required_facts,
            forbid=forbid,
            max_sentences=max_sentences,
        )
        return self._result(text, should_end=should_end, outcome=outcome)

    def _result(self, text: str, *, should_end: bool = False, outcome: Optional[Outcome | str] = None) -> dict[str, Any]:
        # No scripted duplicate repair. If duplicate output happens, ask the LLM once more to vary it.
        if text == self.last_bot_text and not should_end and self.llm_dynamic:
            text = self._generate_reply(
                intent="duplicate_repair",
                user_text="",
                guidance="The previous generated line was accidentally repeated. Say the same business meaning in fresh words and move the call forward.",
                required_facts="Use only the existing account facts and current stage.",
                forbid="Do not repeat the exact previous bot sentence.",
                max_sentences=2,
            )
        self.last_bot_text = text
        if text:
            self.dialogue.append({"speaker": "bot", "text": text})
        if should_end:
            self.closed = True
            self.stage = Stage.CLOSED
        out = outcome.value if isinstance(outcome, Outcome) else outcome
        self._log("bot_reply_final", text=text, should_end=should_end, outcome=out, state=self._state())
        return AgentResult(text, should_end, out, list(self.events), self._state()).to_dict()

    # ------------------------------------------------------------------
    # Dynamic reply helpers: these provide guidance, not full responses.
    # ------------------------------------------------------------------
    def _ask_ptp_days_after_refusal(self, user_text: str) -> str:
        self.stage = Stage.NEGOTIATION
        self.pending_ptp_days = None
        return self._generate_reply(
            intent="ask_ptp_days_after_refusal",
            user_text=user_text,
            guidance="Customer refused same-day payment. Move directly to PTP duration selection. The agent should not ask again if they can pay today.",
            required_facts=f"Ask how many PTP days they need, within 1 to {self.max_ptp_days} days.",
            forbid="Do not ask a binary 'pay today or PTP' question. Do not mention CIBIL here. Do not approve PTP yet.",
            max_sentences=2,
        )

    def _clarify_ptp_duration(self, user_text: str) -> str:
        extra = ""
        if _contains_two_or_three_without_unit(user_text):
            extra = " The transcript contains both two and three, so specifically ask the customer to choose between 2 days and 3 days."
        return self._generate_reply(
            intent="ptp_duration_unclear",
            user_text=user_text,
            guidance="The call is in PTP negotiation, but the exact number of days is unclear. Ask only for a clear PTP duration." + extra,
            required_facts=f"The allowed duration range is 1 to {self.max_ptp_days} days.",
            forbid="Do not ask about today's payment. Do not mention CIBIL. Do not approve PTP yet.",
            max_sentences=2,
        )

    def _no_ptp_message(self, user_text: str = "") -> str:
        return self._generate_reply(
            intent="no_ptp_available",
            user_text=user_text,
            guidance="PTP/extension attempts are exhausted or unavailable. Acknowledge the customer and move to same-day payment or partial payment only.",
            required_facts=f"Due amount is Rs.{self.due_amount}. PTP available is false.",
            forbid="Do not offer any extension, PTP, more days, or future payment date.",
            max_sentences=2,
        )

    def _fine_answer(self, user_text: str) -> str:
        self.last_customer_question = "fine"
        return self._generate_reply(
            intent="answer_fine_late_fee",
            user_text=user_text,
            guidance="Answer the late-fee/fine question honestly. The exact fine amount is not visible, so do not invent a number. Then move the call forward.",
            required_facts=f"Current visible due amount is Rs.{self.due_amount}. Charges may be added as per policy if payment stays pending.",
            forbid="Do not guess the fine amount. Do not threaten.",
            max_sentences=2,
        )

    def _loan_detail_answer(self, user_text: str) -> str:
        self.last_customer_question = "loan_details"
        detail_bits = [f"Loan ID {self.loan_id}", f"due amount Rs.{self.due_amount}"]
        if self.loan_date:
            detail_bits.append(f"loan date {self.loan_date}")
        if self.dealer_name:
            detail_bits.append(f"dealer {self.dealer_name}")
        return self._generate_reply(
            intent="answer_loan_details",
            user_text=user_text,
            guidance="Directly answer the customer's EMI/loan detail question before asking anything else.",
            required_facts="; ".join(detail_bits),
            forbid="Do not add facts not listed. Do not create or confirm PTP in this answer.",
            max_sentences=3,
        )

    def _loan_detail_then_ptp(self, user_text: str, requested_days: Optional[int] = None) -> str:
        if not self.ptp_available():
            self.stage = Stage.NO_PTP
            return self._generate_reply(
                intent="loan_details_then_no_ptp",
                user_text=user_text,
                guidance="Answer loan details first. Then explain that extension is not available and ask for today payment or partial payment.",
                required_facts=f"Loan ID {self.loan_id}; due amount Rs.{self.due_amount}; PTP available false.",
                forbid="Do not offer PTP or more days.",
                max_sentences=3,
            )
        self.stage = Stage.NEGOTIATION
        if requested_days is not None:
            self.pending_ptp_days = requested_days
            return self._generate_reply(
                intent="loan_details_then_ptp_confirm",
                user_text=user_text,
                guidance="Answer loan details first. Then say the requested PTP days are within policy and ask for explicit confirmation before creating it.",
                required_facts=f"Loan ID {self.loan_id}; due amount Rs.{self.due_amount}; requested PTP days {requested_days}; max PTP days {self.max_ptp_days}.",
                forbid="Do not say the PTP is already created. Ask for confirmation only.",
                max_sentences=3,
            )
        self.pending_ptp_days = None
        return self._generate_reply(
            intent="loan_details_then_ptp_duration",
            user_text=user_text,
            guidance="Answer loan details first. Then ask how many PTP days the customer wants.",
            required_facts=f"Loan ID {self.loan_id}; due amount Rs.{self.due_amount}; allowed PTP duration 1 to {self.max_ptp_days} days.",
            forbid="Do not approve PTP yet.",
            max_sentences=3,
        )

    def _amount_answer(self, user_text: str) -> str:
        self.last_customer_question = "amount"
        return self._generate_reply(
            intent="answer_amount",
            user_text=user_text,
            guidance="Answer the amount question directly and then move to the next appropriate recovery step.",
            required_facts=f"Due EMI amount is Rs.{self.due_amount}.",
            forbid="Do not invent extra charges or a different amount.",
            max_sentences=2,
        )

    def _cibil_answer(self, user_text: str) -> str:
        self.last_customer_question = "cibil"
        return self._generate_reply(
            intent="answer_cibil",
            user_text=user_text,
            guidance="Answer the CIBIL/credit-score question calmly. Explain possible policy impact without threatening.",
            required_facts="Payment remaining pending may cause late fee or CIBIL impact as per policy.",
            forbid="Do not guarantee an exact score drop. Do not threaten legal action here.",
            max_sentences=2,
        )

    def _answer_last_question(self, user_text: str) -> str:
        if self.last_customer_question == "fine":
            base = "The customer's previous unanswered topic was fine/late fee."
        elif self.last_customer_question == "loan_details":
            base = "The customer's previous unanswered topic was loan details."
        elif self.last_customer_question == "amount":
            base = "The customer's previous unanswered topic was due amount."
        elif self.last_customer_question == "cibil":
            base = "The customer's previous unanswered topic was CIBIL/credit score."
        else:
            base = "No specific previous question is stored; answer with visible due amount and policy impact only."
        return self._generate_reply(
            intent="answer_first_request",
            user_text=user_text,
            guidance=base + " The customer is insisting that their question be answered before continuing. Answer it briefly, then ask the appropriate next-step question based on the current stage.",
            required_facts=f"Due amount Rs.{self.due_amount}; loan ID {self.loan_id}; PTP available {self.ptp_available()}; max PTP days {self.max_ptp_days}.",
            forbid="Do not ignore the customer's request. Do not create PTP.",
            max_sentences=3,
        )

    def _ask_reason(self, user_text: str) -> str:
        self.reason_asked = True
        self.stage = Stage.REASON
        return self._generate_reply(
            intent="ask_non_payment_reason",
            user_text=user_text,
            guidance="The customer has refused payment more than once. Ask for the real reason payment cannot be made today, so the call can be handled correctly.",
            required_facts=f"Due amount is Rs.{self.due_amount}.",
            forbid="Do not accuse. Do not ask for PTP duration in this turn.",
            max_sentences=2,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> dict[str, Any]:
        self._log("call_started", customer=self._safe_customer_facts(), state=self._state())
        return self._reply_result(
            intent="greeting_identity",
            user_text="__CALL_START__",
            guidance="Start the loan recovery call naturally. Greet as Priya from XYZ Finance and confirm whether you are speaking with the named customer. Do not mention the EMI amount yet.",
            required_facts=f"Customer name: {self.display_name}.",
            forbid="Do not mention due amount, loan ID, fine, CIBIL, PTP, or payment yet.",
            max_sentences=2,
        )

    def stop(self) -> dict[str, Any]:
        self._log("session_stopped_by_user", state=self._state())
        return self._reply_result(
            intent="manual_stop_close",
            user_text="__MANUAL_STOP__",
            guidance="The call is being stopped manually. Close politely and briefly.",
            required_facts="None.",
            forbid="Do not discuss new payment terms.",
            max_sentences=1,
            should_end=True,
            outcome=Outcome.CUSTOMER_ENDED,
        )

    def handle_user_text(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        event_offset = len(self.events)
        if self.closed or self.stage == Stage.CLOSED:
            self._log("input_ignored_closed", text=text[:300], state=self._state())
            return self._result("", should_end=True, outcome=None)

        previous_stage = self.stage.value
        self.lang = _detect_lang(text, self.lang)
        self.turns += 1
        self.dialogue.append({"speaker": "customer", "text": text})
        self._log("user_text_received", text=text[:300], turn=self.turns, detected_lang=self.lang, previous_stage=previous_stage, state=self._state())

        if not text:
            return self._reply_result(
                intent="empty_audio_retry",
                user_text=text,
                guidance="Customer audio was empty or unclear. Ask them to repeat clearly.",
                forbid="Do not mention payment or PTP.",
                max_sentences=1,
            )
        if text == STT_ERROR_TOKEN:
            return self._reply_result(
                intent="stt_error_retry",
                user_text=text,
                guidance="Speech recognition service failed. Apologize briefly and ask the customer to repeat once.",
                forbid="Do not mention technical internals beyond speech issue.",
                max_sentences=1,
            )
        if customer_wants_to_end(text):
            return self._reply_result(
                intent="customer_ending_call",
                user_text=text,
                guidance="Customer wants to end the call. Respectfully close and remind them to clear payment soon.",
                required_facts=f"Due amount Rs.{self.due_amount}.",
                forbid="Do not continue negotiation.",
                max_sentences=1,
                should_end=True,
                outcome=Outcome.CUSTOMER_ENDED,
            )
        if looks_like_background_or_offtopic(text):
            return self._reply_result(
                intent="background_or_offtopic",
                user_text=text,
                guidance="The transcript looks like background/off-topic audio. Ask the customer to repeat their actual answer clearly.",
                forbid="Do not change stage. Do not discuss new loan facts.",
                max_sentences=1,
            )

        # 1) Identity stage
        if self.stage == Stage.IDENTITY:
            if is_wrong_person(text):
                return self._reply_result(
                    intent="wrong_person_close",
                    user_text=text,
                    guidance="Customer says this is the wrong person or wrong number. Apologize and end the call.",
                    forbid="Do not mention EMI details.",
                    max_sentences=1,
                    should_end=True,
                    outcome=Outcome.WRONG_PERSON,
                )
            if is_identity_confirmed(text):
                self.stage = Stage.ASK_PAYMENT
                self._log("stage_transition", from_stage=previous_stage, to_stage=self.stage.value, reason="identity_confirmed")
                return self._reply_result(
                    intent="ask_today_payment",
                    user_text=text,
                    guidance="Identity is confirmed. Inform the customer about the pending EMI and ask if they can clear it today.",
                    required_facts=f"Due amount Rs.{self.due_amount}; customer first name {self.display_first}.",
                    forbid="Do not mention PTP unless customer asks. Do not threaten.",
                    max_sentences=2,
                )
            return self._reply_result(
                intent="identity_clarify",
                user_text=text,
                guidance="Identity is not clear. Ask again only for confirmation that you are speaking with the named customer.",
                required_facts=f"Customer name {self.display_name}.",
                forbid="Do not mention EMI details yet.",
                max_sentences=1,
            )

        # 2) Customer questions always have priority before refusal/reason logic.
        if asks_fine_or_late_fee(text):
            msg = self._fine_answer(text)
            return self._result(msg)
        if asks_loan_details(text):
            requested_days = extract_requested_days(text)
            if mentions_ptp_or_extension(text) or requested_days is not None:
                return self._result(self._loan_detail_then_ptp(text, requested_days))
            return self._result(self._loan_detail_answer(text))
        if asks_amount(text):
            requested_days = extract_requested_days(text)
            if mentions_ptp_or_extension(text) or requested_days is not None:
                if not self.ptp_available():
                    self.stage = Stage.NO_PTP
                    return self._reply_result(
                        intent="amount_then_no_ptp",
                        user_text=text,
                        guidance="Answer the due amount first. Then state that extension is unavailable and ask for same-day payment or partial payment.",
                        required_facts=f"Due amount Rs.{self.due_amount}; PTP available false.",
                        forbid="Do not offer extension.",
                        max_sentences=3,
                    )
                self.stage = Stage.NEGOTIATION
                self._log("stage_transition", from_stage=previous_stage, to_stage=self.stage.value, reason="amount_question_with_ptp")
                if requested_days is not None:
                    self.pending_ptp_days = requested_days
                    return self._reply_result(
                        intent="amount_then_ptp_confirm",
                        user_text=text,
                        guidance="Answer the due amount first. Then ask for explicit confirmation for the requested PTP days if within policy.",
                        required_facts=f"Due amount Rs.{self.due_amount}; requested PTP days {requested_days}; max PTP days {self.max_ptp_days}.",
                        forbid="Do not create PTP yet.",
                        max_sentences=3,
                    )
                return self._reply_result(
                    intent="amount_then_ptp_duration",
                    user_text=text,
                    guidance="Answer the due amount first. Then ask for exact PTP duration.",
                    required_facts=f"Due amount Rs.{self.due_amount}; allowed PTP range 1 to {self.max_ptp_days} days.",
                    forbid="Do not create PTP yet.",
                    max_sentences=3,
                )
            return self._reply_result(
                intent="amount_then_ask_payment",
                user_text=text,
                guidance="Answer the due amount question and ask if the customer can clear it today.",
                required_facts=f"Due amount Rs.{self.due_amount}.",
                forbid="Do not mention PTP unless they ask.",
                max_sentences=2,
            )
        if asks_cibil(text):
            return self._result(self._cibil_answer(text))
        if asks_answer_first(text):
            return self._result(self._answer_last_question(text))

        # 3) Negotiation stage
        if self.stage == Stage.NEGOTIATION:
            requested_days = extract_requested_days(text)
            if requested_days is not None:
                if requested_days <= self.max_ptp_days:
                    self.pending_ptp_days = requested_days
                    self._log("ptp_days_selected", days=requested_days, explicit_confirmation=is_explicit_ptp_confirmation(text))
                    if is_explicit_ptp_confirmation(text):
                        result = self._create_ptp(requested_days)
                        if result["status"] == "success":
                            return self._reply_result(
                                intent="ptp_created_close",
                                user_text=text,
                                guidance="PTP has been created by the system. Inform the customer, mention the deadline, ask them to pay by then, and close politely.",
                                required_facts=f"PTP days {requested_days}; PTP deadline {result['extended_date']}; due amount Rs.{self.due_amount}.",
                                forbid="Do not say it is only being checked; it is already created.",
                                max_sentences=2,
                                should_end=True,
                                outcome=Outcome.PTP_CREATED,
                            )
                        self.stage = Stage.NO_PTP
                        return self._result(self._no_ptp_message(text))
                    return self._reply_result(
                        intent="ptp_days_possible_confirm",
                        user_text=text,
                        guidance="The requested PTP duration is within policy. Ask for explicit confirmation before creating PTP.",
                        required_facts=f"Requested PTP days {requested_days}; max PTP days {self.max_ptp_days}.",
                        forbid="Do not create or say confirmed yet.",
                        max_sentences=2,
                    )
                self.pending_ptp_days = self.max_ptp_days
                return self._reply_result(
                    intent="ptp_days_exceed_max",
                    user_text=text,
                    guidance="Customer requested more days than policy allows. Hold firm on the max allowed duration and ask if they want that max duration noted.",
                    required_facts=f"Requested days {requested_days}; maximum allowed PTP days {self.max_ptp_days}.",
                    forbid="Do not approve more than the max allowed days.",
                    max_sentences=2,
                )

            if is_explicit_ptp_confirmation(text):
                if self.pending_ptp_days is None:
                    return self._reply_result(
                        intent="ptp_confirm_without_days",
                        user_text=text,
                        guidance="Customer confirmed PTP but no exact days are stored. Ask for exact number of days first.",
                        required_facts=f"Allowed PTP duration 1 to {self.max_ptp_days} days.",
                        forbid="Do not create PTP without days.",
                        max_sentences=1,
                    )
                result = self._create_ptp(self.pending_ptp_days)
                if result["status"] == "success":
                    return self._reply_result(
                        intent="ptp_created_close",
                        user_text=text,
                        guidance="PTP has been created by the system. Inform the customer, mention the deadline, ask them to pay by then, and close politely.",
                        required_facts=f"PTP days {self.pending_ptp_days}; PTP deadline {result['extended_date']}; due amount Rs.{self.due_amount}.",
                        forbid="Do not say it is only being checked; it is already created.",
                        max_sentences=2,
                        should_end=True,
                        outcome=Outcome.PTP_CREATED,
                    )
                self.stage = Stage.NO_PTP
                return self._result(self._no_ptp_message(text))

            if _contains_any(text, ("नहीं", "nahi", "no")) and not mentions_ptp_or_extension(text):
                self.refusals += 1
                self._log("refusal_count_incremented", refusals=self.refusals, reason="ptp_negotiation_denial")
                if self.refusals >= 2 and not self.reason_asked:
                    return self._result(self._ask_reason(text))
                return self._reply_result(
                    intent="ptp_denied_ask_partial",
                    user_text=text,
                    guidance="Customer does not want PTP. Ask if any partial payment can be made today.",
                    required_facts=f"Due amount Rs.{self.due_amount}.",
                    forbid="Do not push PTP again in this turn.",
                    max_sentences=2,
                )

            return self._result(self._clarify_ptp_duration(text))

        # 4) PTP / extension request. Guard before payment commitment.
        if mentions_ptp_or_extension(text):
            if not self.ptp_available():
                self.stage = Stage.NO_PTP
                self._log("stage_transition", from_stage=previous_stage, to_stage=self.stage.value, reason="ptp_requested_but_unavailable")
                return self._result(self._no_ptp_message(text))
            requested_days = extract_requested_days(text)
            self.stage = Stage.NEGOTIATION
            self._log("stage_transition", from_stage=previous_stage, to_stage=self.stage.value, reason="ptp_or_extension_requested")
            if requested_days is None:
                self.pending_ptp_days = None
                return self._result(self._clarify_ptp_duration(text))
            if requested_days <= self.max_ptp_days:
                self.pending_ptp_days = requested_days
                if is_explicit_ptp_confirmation(text):
                    result = self._create_ptp(requested_days)
                    if result["status"] == "success":
                        return self._reply_result(
                            intent="ptp_created_close",
                            user_text=text,
                            guidance="PTP has been created by the system. Inform the customer, mention the deadline, ask them to pay by then, and close politely.",
                            required_facts=f"PTP days {requested_days}; PTP deadline {result['extended_date']}; due amount Rs.{self.due_amount}.",
                            forbid="Do not say it is only being checked; it is already created.",
                            max_sentences=2,
                            should_end=True,
                            outcome=Outcome.PTP_CREATED,
                        )
                    self.stage = Stage.NO_PTP
                    return self._result(self._no_ptp_message(text))
                return self._reply_result(
                    intent="ptp_days_possible_confirm",
                    user_text=text,
                    guidance="The requested PTP duration is within policy. Ask for explicit confirmation before creating PTP.",
                    required_facts=f"Requested PTP days {requested_days}; max PTP days {self.max_ptp_days}.",
                    forbid="Do not create or say confirmed yet.",
                    max_sentences=2,
                )
            self.pending_ptp_days = self.max_ptp_days
            return self._reply_result(
                intent="ptp_days_exceed_max",
                user_text=text,
                guidance="Customer requested more days than policy allows. Hold firm on the max allowed duration and ask if they want that max duration noted.",
                required_facts=f"Requested days {requested_days}; maximum allowed PTP days {self.max_ptp_days}.",
                forbid="Do not approve more than the max allowed days.",
                max_sentences=2,
            )

        # 5) Payment commitment
        if is_payment_commitment(text):
            return self._reply_result(
                intent="payment_committed_close",
                user_text=text,
                guidance="Customer committed to payment today. Thank them, remind them of the amount and same-day deadline, and close politely.",
                required_facts=f"Due amount Rs.{self.due_amount}; payment expected today.",
                forbid="Do not ask for PTP.",
                max_sentences=2,
                should_end=True,
                outcome=Outcome.COMMITTED,
            )

        # 6) Reason stage only evaluates real reasons.
        if self.stage == Stage.REASON:
            if not is_clear_reason(text):
                return self._reply_result(
                    intent="reason_unclear_retry",
                    user_text=text,
                    guidance="Customer has not given a clear non-payment reason. Ask for the actual reason payment is stuck today, with examples like salary delay, cash issue, or emergency.",
                    required_facts=f"Due amount Rs.{self.due_amount}.",
                    forbid="Do not ask for PTP duration here.",
                    max_sentences=2,
                )

            self._log("reason_given", text=text[:160])
            if is_critical_reason(text):
                self._log("reason_critical", value=True)
                return self._reply_result(
                    intent="critical_reason_escalate_close",
                    user_text=text,
                    guidance="Customer gave a serious hardship reason such as medical emergency/job loss/accident. Be empathetic, avoid pressure, say the call is being escalated to a senior for review, and close.",
                    required_facts="Senior officer/team will review the situation.",
                    forbid="Do not demand immediate payment in this turn. Do not offer PTP.",
                    max_sentences=2,
                    should_end=True,
                    outcome=Outcome.ESCALATED,
                )

            self._log("reason_critical", value=False)
            self.refusals += 1
            self._log("refusal_count_incremented", refusals=self.refusals, reason="non_critical_reason")
            if self.refusals >= self.max_refusals:
                return self._reply_result(
                    intent="max_refusals_escalate_close",
                    user_text=text,
                    guidance="Customer has crossed max refusal attempts. Calmly say the matter is being forwarded to the senior team for follow-up and close.",
                    forbid="Do not keep negotiating.",
                    max_sentences=2,
                    should_end=True,
                    outcome=Outcome.ESCALATED,
                )
            if self.ptp_available():
                self.stage = Stage.ASK_PAYMENT
                self._log("stage_transition", from_stage=previous_stage, to_stage=self.stage.value, reason="reason_recorded_ptp_available")
                return self._reply_result(
                    intent="reason_recorded_offer_practical_options",
                    user_text=text,
                    guidance="Acknowledge the reason. Ask for practical next step: partial payment today or PTP within policy.",
                    required_facts=f"Due amount Rs.{self.due_amount}; max PTP days {self.max_ptp_days}.",
                    forbid="Do not create PTP yet.",
                    max_sentences=2,
                )
            self.stage = Stage.NO_PTP
            self._log("stage_transition", from_stage=previous_stage, to_stage=self.stage.value, reason="reason_recorded_no_ptp")
            return self._result(self._no_ptp_message(text))

        # 7) Cannot pay/refusal
        if refuses_payment(text):
            self.refusals += 1
            self.push_count += 1
            self._log("refusal_count_incremented", refusals=self.refusals, push_count=self.push_count, reason="payment_refusal")
            if self.refusals >= 2 and not self.reason_asked:
                return self._result(self._ask_reason(text))
            if self.refusals >= self.max_refusals:
                return self._reply_result(
                    intent="max_refusals_escalate_close",
                    user_text=text,
                    guidance="Customer has crossed max refusal attempts. Calmly say the matter is being forwarded to the senior team for follow-up and close.",
                    forbid="Do not keep negotiating.",
                    max_sentences=2,
                    should_end=True,
                    outcome=Outcome.ESCALATED,
                )
            if self.ptp_available():
                return self._result(self._ask_ptp_days_after_refusal(text))
            self.stage = Stage.NO_PTP
            self._log("stage_transition", from_stage=previous_stage, to_stage=self.stage.value, reason="refusal_no_ptp")
            return self._result(self._no_ptp_message(text))

        # 8) Short yes/okay after a prompt
        if _contains_any(text, ("हाँ", "हां", "haan", "yes", "ok", "okay", "ठीक")):
            if self.stage == Stage.ASK_PAYMENT:
                return self._reply_result(
                    intent="short_yes_payment_commitment_close",
                    user_text=text,
                    guidance="Customer gave a short yes after the payment prompt. Treat it as a same-day payment commitment, thank them, mention amount, and close.",
                    required_facts=f"Due amount Rs.{self.due_amount}; payment today.",
                    forbid="Do not ask more questions.",
                    max_sentences=2,
                    should_end=True,
                    outcome=Outcome.COMMITTED,
                )
            if self.stage == Stage.NO_PTP:
                return self._reply_result(
                    intent="short_yes_no_ptp_partial",
                    user_text=text,
                    guidance="Customer said yes while no PTP is available. Clarify that only today payment or partial payment is possible and ask what amount they can arrange now.",
                    required_facts=f"Due amount Rs.{self.due_amount}; PTP unavailable.",
                    forbid="Do not offer extension.",
                    max_sentences=2,
                )

        # 9) Fallback: never repeat the same binary question after resistance.
        if self.ptp_available():
            if self.stage in (Stage.ASK_PAYMENT, Stage.NEGOTIATION) or self.refusals > 0 or self.push_count > 0:
                return self._result(self._ask_ptp_days_after_refusal(text))
            return self._reply_result(
                intent="general_followup",
                user_text=text,
                guidance="Acknowledge the customer and move the call forward with one clear next-step question.",
                required_facts=f"Due amount Rs.{self.due_amount}; PTP available {self.ptp_available()}; max PTP days {self.max_ptp_days}.",
                forbid="Do not sound scripted. Do not ask multiple questions.",
                max_sentences=2,
            )
        self.stage = Stage.NO_PTP
        self._log("stage_transition", from_stage=previous_stage, to_stage=self.stage.value, reason="fallback_no_ptp")
        return self._result(self._no_ptp_message(text))

def load_customers(data_file: Path | str = DATA_FILE) -> list[dict[str, Any]]:
    return json.loads(Path(data_file).read_text(encoding="utf-8"))


def make_session(customer_index: int = 0, data_file: Path | str = DATA_FILE) -> RecoveryAgentSession:
    customers = load_customers(data_file)
    return RecoveryAgentSession(customers[customer_index], all_customers=customers, customer_index=customer_index, data_file=data_file)


if __name__ == "__main__":
    s = make_session(0)
    print("BOT:", s.start()["bot_text"])
    while True:
        u = input("YOU: ").strip()
        if not u:
            continue
        r = s.handle_user_text(u)
        print("BOT:", r["bot_text"])
        print("STATE:", r["state"])
        if r["should_end_call"]:
            break
