import os

# Set FFmpeg path explicitly for Whisper(yaha pa tara path rahega windows wala)
os.environ["PATH"] += os.pathsep + r#"C:\Users\DHANAN~1\Downloads\ffmpeg-2025-09-18-git-c373636f55-full_build\ffmpeg-2025-09-18-git-c373636f55-full_build\bin"

import whisper

import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper

# Load Whisper model ("tiny", "base", "small", "medium", "large")(use small kiya hai prefer medium hai
model = whisper.load_model("small")

def voice_to_text(duration=5, samplerate=16000, filename="temp.wav"):
    """Record voice for `duration` seconds and return transcribed text."""
    print("🎙️ Recording... Speak now!")
    
    # Record audio
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    audio = audio.squeeze()

    # Save as WAV
    sf.write(filename, audio, samplerate, subtype='PCM_16')

    # Transcribe with Whisper
    result = model.transcribe(filename, fp16=False)
    
    return result["text"].strip()

if __name__ == "__main__":
    text = voice_to_text(duration=8)  # record for 5 seconds for tiny,base ka liya 6 s,small ka liya 8s,medium -10s,large -14s
    print("📝 Transcribed:", text)
