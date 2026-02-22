# tts_cli_player_basic.py

# conda activate tortoise
# cd "c:\Users\gauth\OneDrive\Desktop\Cart 498\tortoise-tts"
# $env:PYTHONPATH = "$PWD;$PWD\tortoise"
# pip install sounddevice
# python tts_cli_player_basic.py

import torch
import sounddevice as sd

from tortoise.api_fast import TextToSpeech
from tortoise.utils.audio import load_voices


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

        print("Generating...")
        voice_samples, _ = load_voices([voice])
        audio = tts.tts(
            text,
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
        # Play at 24 kHz
        sd.play(wav.squeeze(0).numpy(), 24000)
        sd.wait()
        print("Done.")


if __name__ == "__main__":
    main()
