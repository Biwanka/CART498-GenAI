# XTTS + Ollama (single narrator voice)
#
# conda activate tortoise
# cd "C:\Users\gauth\OneDrive\Desktop\GitHub\CART498-GenAI\TTS"
# pip install TTS sounddevice
# python tts_cli_player_llm.py

import subprocess
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "en"
LLM_MODEL = "llama3.2:1b"
OLLAMA_EXE = r"C:\Users\gauth\AppData\Local\Programs\Ollama\ollama.exe"

VOICE_DIR = Path(r"C:\Users\gauth\OneDrive\Desktop\GitHub\CART498-GenAI\Tortoise\tortoise\voices")
NARRATOR_FILES = [
    VOICE_DIR / "train_dotrice" / "1.wav",
    VOICE_DIR / "train_dotrice" / "2.wav",
]

SYSTEM_PROMPT = (
    "You are an NPC in a fantasy RPG. Reply with one short sentence under 10 words."
)


def resolve_files(paths):
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        raise FileNotFoundError("No narrator speaker wav files found. Update NARRATOR_FILES.")
    return existing


def run_ollama(prompt):
    cmd = [OLLAMA_EXE, "run", LLM_MODEL, prompt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
    except FileNotFoundError:
        result = subprocess.run(["ollama", "run", LLM_MODEL, prompt], capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
    return result.stdout.strip()


def generate_reply(user_text):
    prompt = f"{SYSTEM_PROMPT}\nPlayer: {user_text}\nNPC:".strip()
    try:
        reply = run_ollama(prompt)
    except subprocess.CalledProcessError as e:
        return f"[Ollama error: {e.stderr.strip()}]"
    if not reply:
        return "[No reply from LLM]"
    return reply.split("\n")[0][:140]


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

        print("Thinking...")
        reply = generate_reply(text)
        print(f"NPC: {reply}")

        print("Speaking...")
        wav = tts.tts(text=reply, speaker_wav=speaker_wavs, language=LANGUAGE)
        sd.play(np.asarray(wav, dtype=np.float32), samplerate=24000)
        sd.wait()
        print("Done.")


if __name__ == "__main__":
    main()

