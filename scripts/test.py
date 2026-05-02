"""
Quick test script for TransLive.
Tests the REST API endpoints without needing a browser or microphone.

Usage:
    1. Start server:  python -m uvicorn app.main:app --port 8766
    2. Run tests:     python scripts/test.py
"""
import sys
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8766"


def request(method, path, data=None):
    url = BASE + path
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"  [FAIL] Cannot connect to {url}: {e}")
        return None


def test_health():
    print("\n=== Health Check ===")
    r = request("GET", "/api/health")
    if not r:
        return False
    print(f"  ASR loaded: {r['asr_loaded']}")
    print(f"  MT loaded:  {r['mt_loaded']}")
    print(f"  Source:     {r['source_lang']} -> Target: {r['target_lang']}")
    return True


def test_translate():
    print("\n=== Translate API ===")
    tests = [
        {"text": "Hello, how are you today?", "source_lang": "en", "target_lang": "zh"},
        {"text": "The weather is nice today.", "source_lang": "en", "target_lang": "zh"},
        {"text": "I would like to book a flight to Tokyo.", "source_lang": "en", "target_lang": "zh"},
    ]
    for t in tests:
        r = request("POST", "/api/translate", t)
        if not r:
            continue
        print(f"  [{t['source_lang']}→{t['target_lang']}] {t['text']}")
        print(f"    => {r['translated']}")


def test_transcribe_file():
    print("\n=== Transcribe File API ===")
    print("  (Skipping — provide a wav/mp3 path to test manually)")
    print("  Example: POST /api/transcribe {\"file_path\": \"test.wav\"}")


def main():
    print("TransLive Test Suite")
    print("=" * 40)

    if not test_health():
        print("\nServer is not running. Start it first:")
        print("  python -m uvicorn app.main:app --port 8766")
        sys.exit(1)

    test_translate()
    test_transcribe_file()
    print("\nDone.")


if __name__ == "__main__":
    main()
