#!/usr/bin/env python3
"""
Test script to verify Medical AI system functionality
"""

def test_emergency_detection():
    """Test emergency keyword detection."""
    print("🧪 Testing Emergency Detection:")
    
    from rag_system import is_medical_emergency, get_emergency_response
    
    test_phrases = [
        "I have chest pain",
        "How do I treat diabetes?", 
        "What is the weather today?",
        "I think I'm having a heart attack",
        "What are the symptoms of high blood pressure?"
    ]
    
    for phrase in test_phrases:
        is_emergency = is_medical_emergency(phrase)
        status = "🚨 EMERGENCY" if is_emergency else "✅ Normal"
        print(f"   \"{phrase}\" -> {status}")
    
    print(f"\n📋 Emergency Response Preview:")
    emergency_resp = get_emergency_response()
    print(f"   {emergency_resp[:100]}...")

def test_rag_system():
    """Test RAG medical knowledge retrieval."""
    print("\n🧪 Testing RAG System:")
    
    try:
        from rag_system import MedicalRAGSystem
        rag = MedicalRAGSystem()
        
        if rag.vectorstore:
            # Test medical queries
            queries = ["diabetes symptoms", "chest pain", "blood pressure"]
            
            for query in queries:
                contexts = rag.retrieve_relevant_context(query, k=2)
                print(f"\n   Query: \"{query}\"")
                print(f"   Found {len(contexts)} relevant contexts")
                
                if contexts:
                    print(f"   Sample: {contexts[0][:80]}...")
        else:
            print("   ⚠️ RAG system not properly initialized")
            
    except Exception as e:
        print(f"   ⚠️ RAG test failed: {e}")

def test_memory_system():
    """Test conversation memory."""
    print("\n🧪 Testing Memory System:")
    
    try:
        from main import MedicalMemoryManager
        memory = MedicalMemoryManager()
        
        print(f"   Loaded {len(memory.conversation_history)} previous conversations")
        
        # Test adding interaction
        memory.add_interaction("Test question", "Test response")
        print("   ✅ Successfully added test interaction")
        
        # Test context generation
        context = memory.get_context_prompt("Current question")
        print(f"   ✅ Generated context prompt ({len(context)} chars)")
        
    except Exception as e:
        print(f"   ⚠️ Memory test failed: {e}")

def test_models():
    """Test model availability."""
    print("\n🧪 Testing Model Availability:")
    
    # Test torch
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"   🔧 CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("   ❌ PyTorch not available")
    
    # Test faster-whisper
    try:
        from faster_whisper import WhisperModel
        print("   ✅ Faster-Whisper available")
    except ImportError:
        print("   ❌ Faster-Whisper not available")
    
    # Test TTS
    try:
        from TTS.api import TTS
        print("   ✅ TTS (Coqui) available")
    except ImportError:
        print("   ❌ TTS not available")
    
    # Test LangChain
    try:
        from langchain_ollama import OllamaLLM
        print("   ✅ LangChain Ollama available")
    except ImportError:
        print("   ❌ LangChain Ollama not available")
    
    # Test audio
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"   ✅ Audio system: {len(devices)} devices found")
    except ImportError:
        print("   ❌ Sounddevice not available")

def main():
    """Run all tests."""
    print("🏥 Medical AI System Tests")
    print("=" * 40)
    
    test_models()
    test_emergency_detection()
    test_rag_system()
    test_memory_system()
    
    print("\n" + "=" * 40)
    print("✅ Testing complete!")
    print("\n🚀 To run the Medical AI Assistant:")
    print("   python main.py              # Enhanced version")
    print("   python main_optimized.py    # Async optimized version")

if __name__ == "__main__":
    main()
