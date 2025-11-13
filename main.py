# main.py - Medical AI Assistant (ULTRA-FAST VERSION)

import os
import random
import wave
import re
import unicodedata
from typing import List, Tuple, Optional
from datetime import datetime
import json

import numpy as np
try:
    import gradio as gr
except Exception as e:
    gr = None
    print(f"[!] gradio not available: {e}")

try:
    from TTS.api import TTS
except Exception as e:
    TTS = None
    print(f"[!] TTS import failed: {e}")

try:
    from faster_whisper import WhisperModel
except Exception as e:
    WhisperModel = None
    print(f"[!] faster_whisper import failed: {e}")

try:
    from langchain_ollama import OllamaLLM
except Exception as e:
    OllamaLLM = None
    print(f"[!] langchain_ollama import failed: {e}")
import time
import threading
import tempfile
import atexit
import concurrent.futures

# ============ IMPORT CONFIGURATION ============
from config import CONFIG, MEDICAL_DATASET, EMERGENCY_CONFIG
from rag_system import MedicalRAGSystem

# ============ MEMORY MANAGER (NO LANGCHAIN!) ============
class MedicalMemoryManager:
    """Simple conversation history - NO LANGCHAIN"""
    
    def __init__(self, memory_file: str = None):
        if memory_file is None:
            memory_file = CONFIG.get("memory_file", "memory.json")
        
        self.memory_file = memory_file
        self.conversation_history = self._load_memory()
        self.max_turns = CONFIG.get("max_memory_turns", 10)
        # Debounce timer and dedicated executor for background saves
        self._save_timer: Optional[threading.Timer] = None
        self._save_delay: float = CONFIG.get("memory_save_delay", 5.0)
        try:
            self._memory_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        except Exception:
            self._memory_executor = None

    def _load_memory(self) -> List[dict]:
        """Load conversation history"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[!] Memory load error: {e}")
        return []

    def _save_memory(self):
        """Debounced async save: schedule a background write after _save_delay seconds."""
        def _write_now(data):
            try:
                with open(self.memory_file, "w", encoding="utf-8") as f:
                    json.dump(list(data), f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[!] Memory background save error: {e}")

        try:
            # cancel previous timer
            if self._save_timer and self._save_timer.is_alive():
                try:
                    self._save_timer.cancel()
                except Exception:
                    pass

            def _delayed():
                if self._memory_executor:
                    try:
                        # pass a snapshot to avoid concurrency issues
                        self._memory_executor.submit(_write_now, list(self.conversation_history))
                    except Exception as e:
                        print(f"[!] Failed to schedule memory write: {e}")
                else:
                    # fallback to immediate write
                    _write_now(list(self.conversation_history))

            self._save_timer = threading.Timer(self._save_delay, _delayed)
            self._save_timer.daemon = True
            self._save_timer.start()
        except Exception as e:
            print(f"[!] Failed to schedule memory save: {e}")

    def flush(self, timeout: float = 10.0):
        """Force immediate write and wait for completion (used at shutdown)."""
        try:
            if self._save_timer and self._save_timer.is_alive():
                try:
                    self._save_timer.cancel()
                except Exception:
                    pass
            # perform synchronous write via executor if available
            if self._memory_executor:
                fut = self._memory_executor.submit(lambda d: open(self.memory_file, "w", encoding="utf-8").write(json.dumps(d, indent=2, ensure_ascii=False)), list(self.conversation_history))
                try:
                    fut.result(timeout=timeout)
                except Exception as e:
                    print(f"[!] Memory flush error: {e}")
            else:
                with open(self.memory_file, "w", encoding="utf-8") as f:
                    json.dump(list(self.conversation_history), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Failed to flush memory: {e}")

    def add(self, user_text: str, ai_text: str):
        """Add conversation turn"""
        now = datetime.now().isoformat()
        self.conversation_history.append({
            "role": "user",
            "content": user_text,
            "timestamp": now
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": ai_text,
            "timestamp": now
        })
        
        # Keep only recent turns
        if len(self.conversation_history) > self.max_turns * 2:
            self.conversation_history = self.conversation_history[-(self.max_turns * 2):]
        
        self._save_memory()

    def context_prompt(self, current_input: str) -> str:
        """Build context with recent history"""
        ctx = ""
        if self.conversation_history:
            ctx = "Recent context:\n"
            for turn in self.conversation_history[-2:]:
                role = "User" if turn["role"] == "user" else "Assistant"
                ctx += f"{role}: {turn['content']}\n"
            ctx += "\n"
        
        return (
            f"Current question: {current_input}\n\n"
            "You are a helpful, concise medical assistant. Be clear and friendly. "
            "Do not provide diagnoses. Encourage consulting professionals.\n\n"
            f"{ctx}Respond to: {current_input}"
        )

# ============ EMERGENCY DETECTION ============
def is_medical_emergency(text: str) -> bool:
    keywords = EMERGENCY_CONFIG.get("keywords", [])
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)

def get_emergency_response() -> str:
    return EMERGENCY_CONFIG.get("priority_response", 
        "🚨 Medical emergency detected. Call emergency services immediately!")

# ============ UTILITY FUNCTIONS ============
def unique_filename(prefix="file", ext=".wav") -> str:
    return f"{prefix}_{random.randint(100000, 999999)}{ext}"

def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS - OPTIMIZED"""
    # Remove unicode combining characters (FIXES character error)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Remove markdown and special chars
    text = re.sub(r'[*\[\]()#_]', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove emojis and special unicode
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Clean whitespace
    text = ' '.join(text.split())
    return text.strip()

# ============ SYSTEM INITIALIZATION ============
print("=" * 60)
print("🏥 SYDNEY MEDICAL AI - ULTRA-FAST")
print("=" * 60)

# Thread pool
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG.get("max_workers", 2))

# Initialize RAG
print("\n[+] Loading knowledge base...")
try:
    rag_system = MedicalRAGSystem()
    print("[✓] Knowledge base ready")
except Exception as e:
    print(f"[!] RAG error: {e}")
    rag_system = None

# Initialize Whisper
print("\n[+] Loading Whisper...")
whisper_model = None
whisper_sizes = ["tiny", "base", "small", "medium", "large"]
preferred_size = CONFIG.get("whisper_model_size", "tiny")  # Default to tiny for speed

if preferred_size not in whisper_sizes:
    print(f"[!] Invalid whisper_model_size '{preferred_size}', falling back to tiny")
    preferred_size = "tiny"

try:
    print(f"[+] Loading Whisper model (size={preferred_size})...")
    try:
        # Try loading preferred size first
        whisper_model = WhisperModel(
            preferred_size,
            device="cpu",
            compute_type="int8",
        )
        print(f"[✓] Whisper ({preferred_size}) ready")
    except Exception as first_error:
        if preferred_size != "tiny":
            # If preferred size fails and it's not tiny, try tiny as fallback
            print(f"[!] Failed to load {preferred_size} model, falling back to tiny...")
            try:
                whisper_model = WhisperModel(
                    "tiny",
                    device="cpu",
                    compute_type="int8",
                )
                print("[✓] Whisper (tiny) ready")
            except Exception as e:
                print(f"[!] Failed to load tiny model: {e}")
                raise
        else:
            raise first_error
except Exception as e:
    print(f"[!] Whisper error: {e} -- continuing without transcription (whisper_model=None)")
    whisper_model = None

# Initialize TTS with MINIMAL logging
print("\n[+] Loading TTS...")
try:
    import logging
    logging.getLogger("TTS").setLevel(logging.ERROR)  # Silence TTS logs
    
    tts = TTS(
        model_name=CONFIG.get("tts_model", "tts_models/en/ljspeech/glow-tts"),
        progress_bar=False,
        gpu=False,  # Force CPU
    )
    print(f"[✓] TTS ready")
except Exception as e:
    print(f"[!] TTS error: {e} -- continuing without TTS (tts=None)")
    tts = None

# Initialize LLM
print("\n[+] Loading LLM...")
try:
    llm = OllamaLLM(
        model="MedGemma-4B",
        temperature=0.7,
        ollama_url="http://127.0.0.1:11434"
    )
    try:
        _ = llm.invoke("Test")
        print("[✓] LLM ready")
    except Exception as e:
        print(f"[!] LLM test invoke failed: {e} -- LLM will be available if Ollama starts")
        llm = llm  # keep the object but calls may fail later
except Exception as e:
    print(f"[!] LLM error: {e} -- continuing without LLM (llm=None)")
    llm = None

# Initialize Memory
print("\n[+] Loading memory...")
memory_manager = MedicalMemoryManager()
print(f"[✓] Memory: {len(memory_manager.conversation_history)} msgs")

_last_llm_response: Optional[str] = None

print("\n" + "=" * 60)
print("✅ READY")
print("=" * 60 + "\n")

# ============ TRANSCRIPTION ============
def transcribe_wav(path: str) -> str:
    if whisper_model is None:
        print("[!] Transcription requested but whisper_model is not available")
        return ""

    start = time.perf_counter()
    try:
        segments, _ = whisper_model.transcribe(
            path,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            vad_filter=False  # Disabled VAD filter temporarily
        )
        text = "".join(seg.text for seg in segments).strip()
        elapsed = time.perf_counter() - start
        print(f"timing: transcribe_wav -> {elapsed:.2f}s")
        return text
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"[!] Transcription error: {e} (after {elapsed:.2f}s)")
        return ""

def transcribe_async(path: str) -> str:
    future = executor.submit(transcribe_wav, path)
    return future.result()

# ============ LLM INTERACTION ============
def ask_llm(user_text: str) -> str:
    global _last_llm_response

    if is_medical_emergency(user_text):
        return get_emergency_response()

    context_prompt = memory_manager.context_prompt(user_text)
    
    if rag_system:
        try:
            enhanced = rag_system.enhance_prompt_with_rag(context_prompt, user_text)
        except Exception as e:
            print(f"[!] RAG error: {e}")
            enhanced = context_prompt
    else:
        enhanced = context_prompt

    prompt_engineered = (
        "You are a medical assistant. Be concise and direct. "
        "No greetings, no meta-text. 3 paragraphs max. "
        "Start with key information. No diagnosis.\n\n"
        "End with one thoughtful follow-up question that deepens understanding of the topic.\n\n"
        f"{enhanced}"
    )

    # If llm is not available, return a safe fallback answer
    if llm is None:
        print("[!] LLM unavailable, returning fallback response")
        fallback = (
            "I cannot access the language model right now. Please try again later.\n\n"
            f"{CONFIG.get('medical_disclaimer', '')}"
        )
        _last_llm_response = fallback
        return fallback

    start = time.perf_counter()
    try:
        resp = llm.invoke(prompt_engineered)

        if _last_llm_response and resp.strip() == _last_llm_response.strip():
            resp = llm.invoke(prompt_engineered + "\n(Rephrase briefly.)")

        # Clean response
        lines = [line.strip() for line in resp.splitlines() if line.strip()]
        
        skip_patterns = [
            "okay", "here's", "hi there", "let's", "aiming",
            "i understand", "i hope", "let me know"
        ]
        
        filtered = []
        for line in lines:
            if not any(p in line.lower() for p in skip_patterns):
                filtered.append(line)
        
        resp = "\n".join(filtered)
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"[!] LLM error: {e} (after {elapsed:.2f}s)")
        return f"⚠️ LLM error: {e}\n\n{CONFIG.get('medical_disclaimer', '')}"

    elapsed = time.perf_counter() - start
    print(f"timing: ask_llm -> {elapsed:.2f}s")

    health_terms = ["health", "medical", "pain", "symptom", "treatment"]
    if any(term in user_text.lower() for term in health_terms):
        if "disclaimer" not in resp.lower():
            resp += f"\n\n{CONFIG.get('medical_disclaimer', '')}"

    _last_llm_response = resp
    return resp

# ============ FAST TTS (OPTIMIZED) ============
def tts_to_file_fast(text: str) -> str:
    """ULTRA-FAST TTS - Single chunk, minimal processing"""
    if tts is None:
        print("[!] TTS requested but tts engine is not available")
        return None

    text = clean_text_for_tts(text)
    # Truncate if too long (sacrifice completeness for speed)
    max_chars = 1000  # MUCH smaller for speed
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    final_wav = unique_filename("response", ".wav")
    start = time.perf_counter()
    try:
        # Single TTS call with speed boost
        tts.tts_to_file(
            text=text,
            file_path=final_wav,
            speed=1.3  # FASTER speech
        )
        elapsed = time.perf_counter() - start
        print(f"timing: tts_to_file_fast -> {elapsed:.2f}s")
        return final_wav
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"[!] TTS error: {e} (after {elapsed:.2f}s)")
        return None

# ============ GRADIO HANDLERS ============
def handle_text(user_text: str) -> Tuple[str, Optional[str]]:
    """Handle text input - FAST"""
    if not user_text or not user_text.strip():
        return "⚠️ Please type something.", None
    
    try:
        answer = ask_llm(user_text)
        memory_manager.add(user_text, answer)
        
        # Start TTS in background (non-blocking)
        audio_future = executor.submit(tts_to_file_fast, answer)
        
        try:
            audio_path = audio_future.result(timeout=30)  # Increased timeout to 30s
        except Exception as e:
            print(f"[!] TTS timeout: {e}")
            audio_path = None
        
        return answer, audio_path
        
    except Exception as e:
        print(f"[!] Error: {e}")
        return f"⚠️ Error: {str(e)}", None

def handle_mic(audio_path: str) -> Tuple[str, Optional[str]]:
    """Handle microphone - FAST"""
    if not audio_path:
        return "⚠️ No audio.", None
    
    try:
        result = transcribe_async(audio_path)
        if not result.strip():
            return "⚠️ Couldn't understand.", None
        
        answer = ask_llm(result)
        memory_manager.add(result, answer)
        
        try:
            audio_out = tts_to_file_fast(answer)
            # Ensure file is fully written before returning
            if audio_out and os.path.exists(audio_out):
                # Small delay to ensure file is completely written
                time.sleep(0.1)
        except Exception as e:
            print(f"[!] TTS error: {e}")
            audio_out = None

        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
            
        return answer, audio_out

    except Exception as e:
        print(f"[!] Error: {e}")
        return f"⚠️ Error: {str(e)}", None

# ============ GRADIO UI ============
if gr is None:
    print("[!] Gradio is not installed; UI will be skipped. Install gradio to use the web interface.")
    demo = None
else:
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🩺 Sydney – Medical AI (Ultra)")
        gr.Markdown(CONFIG.get("medical_disclaimer", ""))

        with gr.Tab("💬 Chat"):
            gr.Progress()  # Progress bar in UI layout
            with gr.Row():
                user_box = gr.Textbox(
                    label="Ask a health question",
                    placeholder="e.g., What causes headaches?",
                    lines=2,
                )
                send_btn = gr.Button("Ask", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=2):
                    reply_box = gr.Textbox(
                        label="Response",
                        lines=12,
                        show_copy_button=True,
                        placeholder="🧘‍♂️ Chill, AI is bigger than the Internet"
                    )
                with gr.Column(scale=1):
                    audio_out = gr.Audio(label="Audio", type="filepath")

            send_btn.click(
                fn=handle_text,
                inputs=user_box,
                outputs=[reply_box, audio_out]
            )

        with gr.Tab("🎤 Voice"):
            gr.Progress()  # Progress bar in UI layout
            with gr.Row():
                mic_in = gr.Audio(
                    type="filepath",
                    label="Record",
                    format="wav",
                    sources=["microphone"],
                    min_length=1,
                    max_length=15,
                )
                mic_btn = gr.Button("Submit", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=2):
                    reply_box2 = gr.Textbox(
                        label="Response",
                        lines=12,
                        show_copy_button=True,
                        placeholder="🧠 Relax — AI is thinking..."
                    )
                with gr.Column(scale=1):
                    audio_out2 = gr.Audio(label="Audio", type="filepath")

            mic_btn.click(
                fn=handle_mic,
                inputs=mic_in,
                outputs=[reply_box2, audio_out2],
                show_progress="minimal"
            )

# ============ MAIN ============
if __name__ == "__main__":
    print("\n🚀 Launching Gradio...\n")
    
    try:
        if demo is not None:
            # attempt to enable queue when available
            try:
                demo.queue()
            except Exception:
                pass

            demo.launch(
                debug=True,  # Enable debug for more info
                share=True,  # Enable public link
                server_name="127.0.0.1",
                server_port=7861,
                quiet=False  # Show full logging
            )
        else:
            print("[!] No UI available (gradio not installed). Exiting after flushing resources.")
    except KeyboardInterrupt:
        print("\n[!] Shutting down...")
    finally:
        # Flush pending memory writes
        try:
            if 'memory_manager' in globals() and memory_manager is not None:
                print("[+] Flushing memory to disk...")
                try:
                    memory_manager.flush()
                except Exception as e:
                    print(f"[!] Memory flush failed: {e}")
                # Shutdown memory executor if present
                try:
                    if hasattr(memory_manager, '_memory_executor') and memory_manager._memory_executor is not None:
                        memory_manager._memory_executor.shutdown(wait=True)
                except Exception:
                    pass
        except Exception:
            pass

        executor.shutdown(wait=True)

        print("[✓] Done!")
