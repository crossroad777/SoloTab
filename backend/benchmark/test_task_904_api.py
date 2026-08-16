"""
backend/benchmark/test_task_904_api.py
======================================
TASK-904: /api/transcribe_midi and /api/refinger API validation
"""

import os
import sys
import json
import pathlib
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("backend"))

from main import app

client = TestClient(app)


def test_api_transcribe_midi():
    midi_path = "outputs/romance_clean.mid"
    with open(midi_path, "rb") as f:
        res = client.post(
            "/api/transcribe_midi",
            files={"file": ("romance_clean.mid", f, "audio/midi")},
            data={"tuning": "standard", "style_profile": "classic"}
        )
    return {
        "status_code": res.status_code,
        "response": res.json()
    }


def test_api_refinger():
    gp5_path = "outputs/task_901_inspection/romance_translated.gp5"
    with open(gp5_path, "rb") as f:
        res = client.post(
            "/api/refinger",
            files={"file": ("romance_translated.gp5", f, "application/octet-stream")},
            data={"tuning": "standard"}
        )
    return {
        "status_code": res.status_code,
        "response": res.json()
    }


def main():
    midi_res = test_api_transcribe_midi()
    refinger_res = test_api_refinger()
    
    out = {
        "task": "TASK-904: Refingering & MIDI UI API Integration",
        "api_transcribe_midi": midi_res,
        "api_refinger": refinger_res
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
