import os
import json
import vosk
import pyaudio
import wave
from typing import Tuple

class VoskSTT:
    def __init__(self, model_path: str = "vosk-model-small-en-us-0.15"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at {model_path}")
        
        vosk.SetLogLevel(-1)  # Reduce logging
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, 16000)
    
    def transcribe_file(self, audio_path: str) -> Tuple[str, str]:
        """Transcribe audio file using Vosk"""
        wf = wave.open(audio_path, 'rb')
        
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Audio file must be WAV format mono PCM 16kHz")
        
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                if 'text' in result:
                    results.append(result['text'])
        
        # Get final result
        final_result = json.loads(self.rec.FinalResult())
        if 'text' in final_result:
            results.append(final_result['text'])
        
        text = ' '.join(results).strip()
        return text, "en"  # Vosk returns English

# Function to replace Whisper transcription
def transcribe_with_vosk(audio_path: str) -> Tuple[str, str]:
    """Drop-in replacement for Whisper transcription"""
    try:
        vosk_stt = VoskSTT()
        return vosk_stt.transcribe_file(audio_path)
    except Exception as e:
        print(f"⚠ Vosk transcription error: {e}")
        return "", "en"
