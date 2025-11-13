import numpy as np
import sounddevice as sd
import threading
import time
from typing import Optional, List

class VoiceActivityDetector:
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_duration: float = 0.1,  # 100ms chunks
                 silence_threshold: float = 0.01,  # Adjust based on your mic
                 min_speech_duration: float = 0.5,  # Minimum speech to start recording
                 silence_duration: float = 2.0):  # Silence time to stop recording
        
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.silence_duration = silence_duration
        
        self.is_recording = False
        self.audio_buffer = []
        self.speech_detected_time = None
        self.last_speech_time = None
        
    def _calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) for audio energy"""
        return np.sqrt(np.mean(audio_chunk**2))
    
    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech"""
        rms = self._calculate_rms(audio_chunk)
        return rms > self.silence_threshold
    
    def start_listening(self) -> Optional[np.ndarray]:
        """
        Start listening and return audio when speech is detected and completed
        Returns: numpy array of recorded audio or None if interrupted
        """
        print("🎧 Listening... (speak when ready)")
        
        self.audio_buffer = []
        self.is_recording = False
        self.speech_detected_time = None
        self.last_speech_time = None
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"⚠ Audio status: {status}")
            
            audio_chunk = indata[:, 0]  # Mono channel
            current_time = time.inputBufferAdcTime
            
            if self._is_speech(audio_chunk):
                if not self.is_recording:
                    # First speech detected
                    if self.speech_detected_time is None:
                        self.speech_detected_time = current_time
                    elif current_time - self.speech_detected_time >= self.min_speech_duration:
                        # Sustained speech detected, start recording
                        self.is_recording = True
                        print("🎙 Recording started...")
                        # Include recent audio chunks
                        recent_chunks = int(self.min_speech_duration * self.sample_rate / self.chunk_size)
                        self.audio_buffer = [audio_chunk] * recent_chunks
                
                if self.is_recording:
                    self.audio_buffer.append(audio_chunk.copy())
                    self.last_speech_time = current_time
            
            else:  # Silence detected
                if self.is_recording and self.last_speech_time:
                    # Check if silence duration exceeded threshold
                    if current_time - self.last_speech_time >= self.silence_duration:
                        print("✅ Recording complete (silence detected)")
                        return  # This will stop the stream
                
                # Reset speech detection if no sustained speech
                if not self.is_recording:
                    self.speech_detected_time = None
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=audio_callback
            ):
                # Keep the stream alive until recording is complete
                while True:
                    if self.is_recording and self.audio_buffer and self.last_speech_time:
                        # Check if we should stop due to silence
                        if time.time() - self.last_speech_time >= self.silence_duration:
                            break
                    time.sleep(0.1)
            
            if self.audio_buffer:
                # Convert to single numpy array
                audio_data = np.concatenate(self.audio_buffer)
                # Convert to int16 format expected by wav files
                audio_data = (audio_data * 32767).astype(np.int16)
                return audio_data
            
        except KeyboardInterrupt:
            print("\n⏹ Recording interrupted")
            return None
        except Exception as e:
            print(f"⚠ Error during recording: {e}")
            return None
        
        return None

# Convenience function for easy integration
def record_until_silence(sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Record audio until silence is detected"""
    detector = VoiceActivityDetector(sample_rate=sample_rate)
    return detector.start_listening()
