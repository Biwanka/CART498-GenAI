# tts_cli_player_llm.py

# Requires Ollama installed: https://ollama.com/download
# Suggested model: ollama pull llama3.2:3b
# 
# conda activate tortoise
# cd "c:\Users\gauth\OneDrive\Desktop\Cart 498\tortoise-tts"
# $env:PYTHONPATH = "$PWD;$PWD\tortoise"
# pip install sounddevice
# python tts_cli_player_llm.py

import subprocess
import torch
import sounddevice as sd

from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_voices

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are an NPC in a fantasy RPG. Reply with 1 short sentence, "
    "under 12 words."
)


def generate_with_ollama(user_text: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\nPlayer: {user_text}\nNPC:".strip()
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return "[Ollama not installed. Please install Ollama to use LLM mode.]"
    except subprocess.CalledProcessError as e:
        return f"[Ollama error: {e.stderr.strip()}]"

    reply = result.stdout.strip()
    if not reply:
        return "[No reply from LLM]"
    # Keep it short for TTS speed
    return reply.split("\n")[0][:200]


def main():
    # Load model once
    tts = TextToSpeech(use_deepspeed=False, kv_cache=True, half=True)
    voice = "train_dotrice"

    print(f"CUDA available: {torch.cuda.is_available()}")
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
        reply = generate_with_ollama(text)
        print(f"NPC: {reply}")

        print("Speaking...")
        voice_samples, _ = load_voices([voice])
        audio = tts.tts(
            reply,
            voice_samples=voice_samples,
            k=1,
            verbose=False,
            # Speed-friendly settings for demos
            num_autoregressive_samples=32,
            top_p=0.8,
            temperature=0.8,
            max_mel_tokens=200,
        )

        if audio.dim() == 3:
            wav = audio[0].cpu()
        elif audio.dim() == 2:
            wav = audio.cpu()
        else:
            wav = audio.unsqueeze(0).cpu()
        sd.play(wav.squeeze(0).numpy(), 24000)
        sd.wait()
        print("Done.")


if __name__ == "__main__":
    main()
