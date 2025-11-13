# temp_main.py

import os
import json
import random
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import gradio as gr
import soundfile as sf  # for saving mic audio to wav

from TTS.api import TTS
from faster_whisper import WhisperModel
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferWindowMemory

# Your RAG file must be in the same folder
from rag_system import MedicalRAGSystem, is_medical_emergency, get_emergency_response