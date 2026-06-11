import os
import sys
import json
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_model_name():
    try:
        from config import GEMINI_MODEL
        return GEMINI_MODEL
    except Exception:
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = get_model_name()


def pass_msg(name, msg):
    print(f"[PASS] {name}: {msg}")


def fail_msg(name, msg):
    print(f"[FAIL] {name}: {msg}")


def check_env():
    print("=== GEMINI API CHECK START ===")
    print(f"GEMINI_API_KEY present: {bool(GEMINI_API_KEY)}")
    print(f"Model: {MODEL}")
    print()

    if not GEMINI_API_KEY:
        fail_msg("Environment", "GEMINI_API_KEY missing. Add it to .env or system environment.")
        sys.exit(1)

    pass_msg("Environment", "GEMINI_API_KEY found")


def check_gemini_sdk():
    name = "Gemini SDK"

    try:
        from google import genai
    except Exception as e:
        fail_msg(name, f"google-genai not installed: {e}")
        print("Install it with: pip install -U google-genai")
        return

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=MODEL,
            contents="Reply with only this exact text: GEMINI_OK"
        )

        text = (response.text or "").strip()
        if "GEMINI_OK" in text:
            pass_msg(name, text)
        else:
            pass_msg(name, f"API responded, but unexpected text: {text[:300]}")

    except Exception as e:
        fail_msg(name, str(e))


def check_gemini_rest():
    name = "Gemini REST"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Reply with only this exact text: GEMINI_REST_OK"
                    }
                ]
            }
        ]
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)

        if r.status_code != 200:
            fail_msg(name, f"HTTP {r.status_code}: {r.text[:1000]}")
            return

        data = r.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

        if "GEMINI_REST_OK" in text:
            pass_msg(name, text)
        else:
            pass_msg(name, f"API responded, but unexpected text: {text[:300]}")

    except Exception as e:
        fail_msg(name, str(e))


if __name__ == "__main__":
    check_env()
    check_gemini_sdk()
    check_gemini_rest()
    print()
    print("=== GEMINI API CHECK DONE ===")