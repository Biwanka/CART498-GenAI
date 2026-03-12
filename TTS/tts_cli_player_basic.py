# XTTS basic live reader
#
# conda activate tortoise
# cd "C:\Users\gauth\OneDrive\Desktop\GitHub\CART498-GenAI\TTS"
# pip install TTS sounddevice
# python tts_cli_player_basic.py

from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "en"

VOICE_DIR = Path(r"C:\Users\gauth\OneDrive\Desktop\GitHub\CART498-GenAI\Tortoise\tortoise\voices")
NARRATOR_FILES = [
    VOICE_DIR / "train_dotrice" / "1.wav",
    VOICE_DIR / "train_dotrice" / "2.wav",
]


def resolve_files(paths):
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        raise FileNotFoundError("No narrator speaker wav files found. Update NARRATOR_FILES.")
    return existing


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")

    tts = TTS(MODEL_NAME, progress_bar=False).to(device)
    speaker_wavs = resolve_files(NARRATOR_FILES)

    print("Type text and press Enter. Type 'quit' to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            print("Exiting.")
            break

        print("Generating...")
        wav = tts.tts(text=text, speaker_wav=speaker_wavs, language=LANGUAGE)
        sd.play(np.asarray(wav, dtype=np.float32), samplerate=24000)
        sd.wait()
        print("Done.")


if __name__ == "__main__":
    main()
