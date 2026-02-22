# tts_cli_player_llm_pick.py
#
# Choose a voice each time via a numbered menu.
# Requires Ollama installed: https://ollama.com/download
# Suggested model: ollama pull llama3.2:1b
#
# conda activate tortoise
# cd "C:\Users\gauth\OneDrive\Desktop\GitHub\CART498-GenAI\Tortoise"
# $env:PYTHONPATH = "$PWD;$PWD\tortoise"
# pip install sounddevice
# python tts_cli_player_llm_pick.py

import subprocess
import torch
import sounddevice as sd

from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_voices

MODEL = "llama3.2:1b"
OLLAMA_EXE = r"C:\Users\gauth\AppData\Local\Programs\Ollama\ollama.exe"

VOICE_LIST = [
    ("Narrator", "train_dotrice"),
    ("Guard", "train_empire"),
    ("Healer", "train_grace"),
    ("Merchant", "train_lescault"),
]

SYSTEM_PROMPT = (
    "You are an NPC in a fantasy RPG. Reply with 1 short sentence, "
    "under 8 words."
)


def generate_with_ollama(user_text: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\nPlayer: {user_text}\nNPC:".strip()
    try:
        result = subprocess.run(
            [OLLAMA_EXE, "run", MODEL, prompt],
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
    return reply.split("\n")[0][:120]


def pick_voice():
    print("Choose a voice:")
    for i, (label, voice) in enumerate(VOICE_LIST, start=1):
        print(f"{i}. {label} ({voice})")
    while True:
        choice = input("Voice #> ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(VOICE_LIST):
            return VOICE_LIST[int(choice) - 1]
        print("Invalid choice. Try again.")


def main():
    tts = TextToSpeech(use_deepspeed=False, kv_cache=True, half=True)
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

        label, voice = pick_voice()
        print(f"Selected: {label} | Voice: {voice}")

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
