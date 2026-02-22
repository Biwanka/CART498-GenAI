# web_tortoise_ui.py



# $env:PYTHONPATH = "$PWD;$PWD\tortoise"
# python web_tortoise_ui.py


import os
import tempfile

import gradio as gr
import torch
import torchaudio

from tortoise.api import TextToSpeech as QualityTTS
from tortoise.api_fast import TextToSpeech as FastTTS
from tortoise.utils.audio import load_voices

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tts_quality = QualityTTS(use_deepspeed=False, kv_cache=True, half=True, device=DEVICE)
tts_fast = FastTTS(use_deepspeed=False, kv_cache=True, half=True, device=DEVICE)
voice = "train_dotrice"

def synth(text, backend, preset):
    if not text.strip():
        return None
    voice_samples, _ = load_voices([voice])
    if backend == "quality":
        audio = tts_quality.tts_with_preset(
            text,
            preset=preset,
            voice_samples=voice_samples,
            k=1,
            verbose=False,
        )
    else:
        audio = tts_fast.tts(
            text,
            voice_samples=voice_samples,
            k=1,
            verbose=False,
        )
    # Normalize output shape to (1, S)
    if audio.dim() == 3:
        wav = audio[0].cpu()
    elif audio.dim() == 2:
        wav = audio.cpu()
    else:
        wav = audio.unsqueeze(0).cpu()
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="tortoise_")
    os.close(fd)
    torchaudio.save(path, wav, 24000)
    return path

demo = gr.Interface(
    fn=synth,
    inputs=[
        gr.Textbox(lines=3, placeholder="Type a line..."),
        gr.Dropdown(choices=["quality", "fast"], value="fast", label="Backend"),
        gr.Dropdown(choices=["fast", "standard", "high_quality"], value="standard", label="Preset"),
    ],
    outputs=gr.Audio(type="filepath"),
    title="Tortoise TTS (Local)",
    description=f"Device: {DEVICE}. Use Quality for cleaner audio (slower).",
)

demo.launch(share=False, show_api=False, server_name="127.0.0.1")
