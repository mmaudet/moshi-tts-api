#!/usr/bin/env python3
"""
Test all available voices in the Moshi TTS API
Checks which voices work and which return errors
"""

import requests
import json
import sys
from typing import Dict, List, Tuple
from datetime import datetime

API_URL = "http://localhost:8000/api/v1/tts"
TEST_TEXT_FR = "Bonjour, ceci est un test."
TEST_TEXT_EN = "Hello, this is a test."

# All voices from app.py
VOICES = {
    "default": {"name": "default", "lang": "fr", "category": "Default"},

    # VCTK voices (English)
    "vctk/p225_023.wav": {"name": "vctk_p225", "lang": "en", "category": "VCTK English"},
    "vctk/p226_023.wav": {"name": "vctk_p226", "lang": "en", "category": "VCTK English"},
    "vctk/p227_023.wav": {"name": "vctk_p227", "lang": "en", "category": "VCTK English"},
    "vctk/p228_023.wav": {"name": "vctk_p228", "lang": "en", "category": "VCTK English"},
    "vctk/p229_023.wav": {"name": "vctk_p229", "lang": "en", "category": "VCTK English"},
    "vctk/p230_023.wav": {"name": "vctk_p230", "lang": "en", "category": "VCTK English"},
    "vctk/p231_023.wav": {"name": "vctk_p231", "lang": "en", "category": "VCTK English"},
    "vctk/p232_023.wav": {"name": "vctk_p232", "lang": "en", "category": "VCTK English"},
    "vctk/p233_023.wav": {"name": "vctk_p233", "lang": "en", "category": "VCTK English"},
    "vctk/p234_023.wav": {"name": "vctk_p234", "lang": "en", "category": "VCTK English"},

    # CML-TTS French voices
    "cml-tts/fr/1406_1028_000009-0003.wav": {"name": "cml_fr_1406", "lang": "fr", "category": "CML-TTS French"},
    "cml-tts/fr/1591_1028_000108-0004.wav": {"name": "cml_fr_1591", "lang": "fr", "category": "CML-TTS French"},
    "cml-tts/fr/1770_1028_000036-0002.wav": {"name": "cml_fr_1770", "lang": "fr", "category": "CML-TTS French"},
    "cml-tts/fr/2114_1656_000053-0001.wav": {"name": "cml_fr_2114", "lang": "fr", "category": "CML-TTS French"},
    "cml-tts/fr/2154_2576_000020-0003.wav": {"name": "cml_fr_2154", "lang": "fr", "category": "CML-TTS French"},
    "cml-tts/fr/2216_1745_000007-0001.wav": {"name": "cml_fr_2216", "lang": "fr", "category": "CML-TTS French"},
    "cml-tts/fr/2223_1745_000009-0002.wav": {"name": "cml_fr_2223", "lang": "fr", "category": "CML-TTS French"},
    "cml-tts/fr/2465_1943_000152-0002.wav": {"name": "cml_fr_2465", "lang": "fr", "category": "CML-TTS French"},
    "cml-tts/fr/296_1028_000022-0001.wav": {"name": "cml_fr_296", "lang": "fr", "category": "CML-TTS French"},
    "cml-tts/fr/3267_1902_000075-0001.wav": {"name": "cml_fr_3267", "lang": "fr", "category": "CML-TTS French"},

    # Expresso voices (English with emotions)
    "expresso/ex01-ex02_default_001_channel1_168s.wav": {"name": "expresso_default", "lang": "en", "category": "Expresso Emotions"},
    "expresso/ex01-ex02_enunciated_001_channel1_432s.wav": {"name": "expresso_enunciated", "lang": "en", "category": "Expresso Emotions"},
    "expresso/ex01-ex02_fast_001_channel1_104s.wav": {"name": "expresso_fast", "lang": "en", "category": "Expresso Emotions"},
    "expresso/ex01-ex02_projected_001_channel1_46s.wav": {"name": "expresso_projected", "lang": "en", "category": "Expresso Emotions"},
    "expresso/ex01-ex02_whisper_001_channel1_579s.wav": {"name": "expresso_whisper", "lang": "en", "category": "Expresso Emotions"},
    "expresso/ex03-ex01_angry_001_channel1_201s.wav": {"name": "expresso_angry", "lang": "en", "category": "Expresso Emotions"},
    "expresso/ex03-ex01_happy_001_channel1_334s.wav": {"name": "expresso_happy", "lang": "en", "category": "Expresso Emotions"},
    "expresso/ex03-ex01_calm_001_channel1_1143s.wav": {"name": "expresso_calm", "lang": "en", "category": "Expresso Emotions"},
    "expresso/ex03-ex01_confused_001_channel1_909s.wav": {"name": "expresso_confused", "lang": "en", "category": "Expresso Emotions"},

    # EARS voices (English)
    "ears/p001/freeform_speech_01.wav": {"name": "ears_p001", "lang": "en", "category": "EARS English"},
    "ears/p002/freeform_speech_01.wav": {"name": "ears_p002", "lang": "en", "category": "EARS English"},
    "ears/p003/freeform_speech_01.wav": {"name": "ears_p003", "lang": "en", "category": "EARS English"},
    "ears/p004/freeform_speech_01.wav": {"name": "ears_p004", "lang": "en", "category": "EARS English"},
    "ears/p005/freeform_speech_01.wav": {"name": "ears_p005", "lang": "en", "category": "EARS English"},
    "ears/p010/freeform_speech_01.wav": {"name": "ears_p010", "lang": "en", "category": "EARS English"},
    "ears/p015/freeform_speech_01.wav": {"name": "ears_p015", "lang": "en", "category": "EARS English"},
    "ears/p020/freeform_speech_01.wav": {"name": "ears_p020", "lang": "en", "category": "EARS English"},
    "ears/p025/freeform_speech_01.wav": {"name": "ears_p025", "lang": "en", "category": "EARS English"},
    "ears/p030/freeform_speech_01.wav": {"name": "ears_p030", "lang": "en", "category": "EARS English"},
}

def test_voice(voice: str, info: Dict) -> Tuple[bool, str, int]:
    """
    Test a single voice
    Returns: (success, error_message, audio_size)
    """
    text = TEST_TEXT_FR if info["lang"] == "fr" else TEST_TEXT_EN

    payload = {
        "text": text,
        "language": info["lang"],
        "format": "wav",
        "voice": voice
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            audio_size = len(response.content)
            return (True, "", audio_size)
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}"
            return (False, error_msg, 0)
    except requests.exceptions.Timeout:
        return (False, "Timeout (30s)", 0)
    except Exception as e:
        return (False, str(e), 0)

def main():
    print("=" * 80)
    print("🎙️  Moshi TTS Voice Testing")
    print("=" * 80)
    print(f"Testing {len(VOICES)} voices...")
    print(f"API URL: {API_URL}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    results = {
        "success": [],
        "failed": []
    }

    # Group by category
    by_category = {}
    for voice, info in VOICES.items():
        category = info["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((voice, info))

    # Test each category
    for category in sorted(by_category.keys()):
        print(f"\n📂 {category}")
        print("-" * 80)

        voices_in_category = by_category[category]
        for voice, info in voices_in_category:
            print(f"  Testing {info['name']:30} ({info['lang']})... ", end="", flush=True)

            success, error, size = test_voice(voice, info)

            if success:
                print(f"✅ OK ({size:,} bytes)")
                results["success"].append({
                    "voice": voice,
                    "name": info["name"],
                    "category": category,
                    "lang": info["lang"],
                    "size": size
                })
            else:
                print(f"❌ FAILED: {error}")
                results["failed"].append({
                    "voice": voice,
                    "name": info["name"],
                    "category": category,
                    "lang": info["lang"],
                    "error": error
                })

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    total = len(VOICES)
    success_count = len(results["success"])
    failed_count = len(results["failed"])

    print(f"Total voices tested: {total}")
    print(f"✅ Success: {success_count} ({success_count/total*100:.1f}%)")
    print(f"❌ Failed:  {failed_count} ({failed_count/total*100:.1f}%)")

    if results["failed"]:
        print("\n🚨 Failed Voices:")
        print("-" * 80)

        # Group failures by error message
        by_error = {}
        for item in results["failed"]:
            error = item["error"]
            if error not in by_error:
                by_error[error] = []
            by_error[error].append(f"{item['name']} ({item['category']})")

        for error, voices in by_error.items():
            print(f"\n  Error: {error}")
            for voice in voices:
                print(f"    - {voice}")

    if results["success"]:
        print("\n✨ Working Voices by Category:")
        print("-" * 80)

        by_cat = {}
        for item in results["success"]:
            cat = item["category"]
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(item["name"])

        for cat in sorted(by_cat.keys()):
            print(f"\n  {cat}: {len(by_cat[cat])} voices")
            for name in sorted(by_cat[cat]):
                print(f"    - {name}")

    print("\n" + "=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Return exit code based on results
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
