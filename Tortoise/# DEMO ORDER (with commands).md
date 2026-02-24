# DEMO ORDER (with commands)

## 0) SETUP (run once)

conda activate tortoise
cd "C:\Users\gauth\OneDrive\Desktop\GitHub\CART498-GenAI\Tortoise"

### If PowerShell

$env:PYTHONPATH = "$PWD;$PWD\tortoise"

### If cmd

set PYTHONPATH=%CD%;%CD%\tortoise

## 1) OFFLINE WAV SHOWCASE (8‑second clips)

### 1a) Prompt redaction (emotion control)

python tortoise\do_tts.py --text "[I am terrified,] We should not go in there. The air is wrong, the stones are cold, and I swear the walls are whispering our names." --voice train_empire --preset fast --candidates 1 --output_path results\redaction

### 1b) Voice mix (latent blending)

python tortoise\do_tts.py --text "The council will hear your case now, but choose your words carefully, speak with respect, and do not test their patience." --voice "train_dotrice&train_lescault" --preset fast --candidates 1 --output_path results\voice_mix

### 1c) Knobs comparison (speed vs quality)

python tortoise\do_tts.py --text "At dawn we march across the valley, and by dusk the gates will open to us, if our courage does not fail." --voice train_dotrice --preset ultra_fast --candidates 1 --output_path results\knobs
python tortoise\do_tts.py --text "At dawn we march across the valley, and by dusk the gates will open to us, if our courage does not fail." --voice train_dotrice --preset high_quality --candidates 1 --output_path results\knobs

## 2) LIVE BASIC (manual input → Tortoise)

python tts_cli_player_basic.py

(type lines live)

## 3) LLM + TTS (your input → AI response → Tortoise speaks)

python tts_cli_player_llm.py

 (type lines live)

## 4) MULTI‑NPC VOICES (role-based)

python tts_cli_player_llm_roles.py

 (type: merchant: hello, guard: halt, healer: help me)

## 5) OPTIONAL: VOICE PICKER

python tts_cli_player_llm_pick.py

## DEMO ORDER (15‑minute presentation)

## 1) START WITH OFFLINE WAV SHOWCASE (8‑second clips)

Purpose: show Tortoise’s quality + range using its intended workflow

- redaction (emotion control)

- voice mix (latent blending)

- knobs (quality vs speed)

This proves what Tortoise is “known for”: high‑quality offline WAV generation

## 2) BASIC LIVE TTS (manual input → Tortoise)

Script: tts_cli_player_basic.py

Purpose: show how Tortoise can be used interactively in a project

Input: you type a line, it speaks it

This is the “baseline live integration.”

## 3) LLM + TTS

Script: tts_cli_player_llm.py

Purpose: show how Tortoise can be connected to a text‑generating AI (Ollama)

Input: you type a line, LLM responds, Tortoise speaks the response

This demonstrates NPC‑style dialogue generation

## 4) MULTI‑NPC VOICES (role-based voices)

Script: tts_cli_player_llm_roles.py

Purpose: show how different NPCs can have different voices

Input format: "merchant: hello" / "guard: halt"

This demonstrates character identity + voice mapping

## 5) OPTIONAL: VOICE PICKER VERSION

Script: tts_cli_player_llm_pick.py

Purpose: show manual voice selection per line

## NOTES FOR TALKING POINTS

- Offline WAVs = best quality but slower

- Live demo = faster but lower fidelity (fewer samples/shorter output)

- NVIDIA GPU required for usable speed

- No paid API for Tortoise itself; only if using external LLM APIs
