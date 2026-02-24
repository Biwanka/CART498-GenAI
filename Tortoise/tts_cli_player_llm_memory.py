# tts_cli_player_llm_memory.py
#
# LLM + TTS with short-term memory (last 3 turns)
# Requires Ollama installed: https://ollama.com/download
# Suggested model: ollama pull llama3.2:1b
#
# conda activate tortoise
# cd "C:\Users\gauth\OneDrive\Desktop\GitHub\CART498-GenAI\Tortoise"
# $env:PYTHONPATH = "$PWD;$PWD\tortoise"
# pip install sounddevice
# python tts_cli_player_llm_memory.py

import subprocess
from collections import deque

import torch
import sounddevice as sd

from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_voices

MODEL = "llama3.2:1b"
OLLAMA_EXE = r"C:\Users\gauth\AppData\Local\Programs\Ollama\ollama.exe"

VOICE = "train_dotrice"

SYSTEM_PROMPT = (
    "You are an NPC in a fantasy RPG. Reply with 1 short sentence, "
    "under 10 words. Stay in character."
)

MEMORY_TURNS = 3
history = deque(maxlen=MEMORY_TURNS)


def build_prompt(user_text: str) -> str:
    lines = [SYSTEM_PROMPT, "Conversation:"]
    for turn in history:
        lines.append(turn)
    lines.append(f"Player: {user_text}")
    lines.append("NPC:")
    return "\n".join(lines).strip()


def generate_with_ollama(user_text: str) -> str:
    prompt = build_prompt(user_text)
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
    return reply.split("\n")[0][:140]


def main():
    tts = TextToSpeech(use_deepspeed=False, kv_cache=True, half=True)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Memory mode ON (last 3 turns). Type 'quit' to exit.")

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

        # Update memory
        history.append(f"Player: {text}")
        history.append(f"NPC: {reply}")

        print("Speaking...")
        voice_samples, _ = load_voices([VOICE])
        audio = tts.tts(
            reply,
            voice_samples=voice_samples,
            k=1,
            verbose=False,
            num_autoregressive_samples=32,
            top_p=0.8,
            temperature=0.8,
            max_mel_tokens=220,
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
