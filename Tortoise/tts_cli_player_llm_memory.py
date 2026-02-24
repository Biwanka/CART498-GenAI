# tts_cli_player_llm_memory.py

# What it does:

# Keeps the last 3 turns of conversation
# Feeds them into Ollama
# NPC replies with context (“memory”)
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
from collections import deque, defaultdict

import torch
import sounddevice as sd

from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_voices

MODEL = "llama3.2:1b"
OLLAMA_EXE = r"C:\Users\gauth\AppData\Local\Programs\Ollama\ollama.exe"

VOICE_MAP = {
    "narrator": "train_dotrice",
    "merchant": "train_lescault",
    "guard": "train_empire",
    "healer": "train_grace",
}

SYSTEM_PROMPT = (
    "You are an NPC in a fantasy RPG. Reply with 1 short sentence, "
    "under 10 words. Stay in character."
)

MEMORY_TURNS = 3
history_by_role = defaultdict(lambda: deque(maxlen=MEMORY_TURNS * 2))
quest_state_by_role = defaultdict(lambda: "not started")


def parse_role_and_text(raw: str):
    if ":" in raw:
        role, text = raw.split(":", 1)
        return role.strip().lower(), text.strip()
    return "narrator", raw.strip()


def update_quest_state(role: str, user_text: str):
    text = user_text.lower()
    if any(k in text for k in ["quest", "help me", "job", "task"]):
        quest_state_by_role[role] = "started"
    if any(k in text for k in ["completed", "done", "finished"]):
        quest_state_by_role[role] = "completed"


def build_prompt(role: str, user_text: str) -> str:
    state = quest_state_by_role[role]
    lines = [
        SYSTEM_PROMPT,
        f"NPC role: {role}. Quest state: {state}.",
        "Conversation:",
    ]
    for turn in history_by_role[role]:
        lines.append(turn)
    lines.append(f"Player: {user_text}")
    lines.append("NPC:")
    return "\n".join(lines).strip()


def generate_with_ollama(role: str, user_text: str) -> str:
    prompt = build_prompt(role, user_text)
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
    print("Memory mode ON (last 3 turns per role). Type 'quit' to exit.")
    print("Use 'role: message' (e.g., 'merchant: hello').")
    print("Commands: /reset (all), /reset <role>")
    print(f"Roles: {', '.join(VOICE_MAP.keys())}")

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
        if text.startswith("/reset"):
            parts = text.split()
            if len(parts) == 1:
                history_by_role.clear()
                quest_state_by_role.clear()
                print("All memory cleared.")
            else:
                role = parts[1].lower()
                history_by_role.pop(role, None)
                quest_state_by_role.pop(role, None)
                print(f"Memory cleared for role: {role}")
            continue

        role, user_text = parse_role_and_text(text)
        update_quest_state(role, user_text)
        voice = VOICE_MAP.get(role, VOICE_MAP["narrator"])

        print(f"Role: {role} | Voice: {voice} | Quest: {quest_state_by_role[role]}")
        print("Thinking...")
        reply = generate_with_ollama(role, user_text)
        print(f"NPC: {reply}")

        # Update memory
        history_by_role[role].append(f"Player: {user_text}")
        history_by_role[role].append(f"NPC: {reply}")

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
