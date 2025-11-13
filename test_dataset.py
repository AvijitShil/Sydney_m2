"""
Test Dataset Structure for LangChain RAG System
This script verifies that the dataset specified in config.py exists and can be loaded properly.
"""

import os
import sys
from pathlib import Path

from config import CONFIG, MEDICAL_DATASET

def test_dataset_exists():
    """Test if the dataset file exists."""
    dataset_path = CONFIG.get("dataset_path")
    if not dataset_path:
        print("[!] ERROR: No dataset_path specified in CONFIG")
        return False
    
    if not os.path.exists(dataset_path):
        print(f"[!] ERROR: Dataset file not found at: {dataset_path}")
        return False
    
    print(f"[✓] Dataset file exists: {dataset_path}")
    return True

def test_dataset_type():
    """Test if the dataset type is supported."""
    dataset_type = CONFIG.get("dataset_type", "").lower()
    supported_types = ["pdf", "txt", "json"]
    
    if not dataset_type:
        print("[!] ERROR: No dataset_type specified in CONFIG")
        return False
    
    if dataset_type not in supported_types:
        print(f"[!] ERROR: Unsupported dataset type: {dataset_type}")
        print(f"[i] Supported types: {', '.join(supported_types)}")
        return False
    
    print(f"[✓] Dataset type is supported: {dataset_type}")
    return True

def test_dataset_loading():
    """Test if the dataset can be loaded based on its type."""
    dataset_path = CONFIG.get("dataset_path")
    dataset_type = CONFIG.get("dataset_type", "").lower()
    
    if not dataset_path or not dataset_type:
        return False
    
    try:
        if dataset_type == "pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(dataset_path)
            docs = loader.load()
            print(f"[✓] Successfully loaded PDF: {len(docs)} pages")
            
            if docs:
                print(f"[i] First page preview: {docs[0].page_content[:100]}...")
            
        elif dataset_type == "txt":
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(dataset_path)
            docs = loader.load()
            print(f"[✓] Successfully loaded TXT: {len(docs)} documents")
            
            if docs:
                print(f"[i] Content preview: {docs[0].page_content[:100]}...")
            
        elif dataset_type == "json":
            from langchain_community.document_loaders import JSONLoader
            loader = JSONLoader(
                file_path=dataset_path,
                jq_schema=".",
                text_content=False
            )
            docs = loader.load()
            print(f"[✓] Successfully loaded JSON: {len(docs)} documents")
            
            if docs:
                print(f"[i] Content preview: {docs[0].page_content[:100]}...")
        
        return True
    
    except Exception as e:
        print(f"[!] ERROR loading dataset: {e}")
        return False

def test_embedding_model():
    """Test if the embedding model can be loaded."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embedding_model_name = CONFIG.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"[+] Testing embedding model: {embedding_model_name}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test with a simple text
        test_text = "This is a test sentence for embeddings."
        embedding = embeddings.embed_query(test_text)
        
        print(f"[✓] Embedding model loaded successfully")
        print(f"[i] Embedding dimension: {len(embedding)}")
        return True
    
    except Exception as e:
        print(f"[!] ERROR loading embedding model: {e}")
        return False

def run_all_tests():
    """Run all tests and return overall success."""
    print("\n" + "="*60)
    print("TESTING DATASET FOR LANGCHAIN RAG SYSTEM")
    print("="*60 + "\n")
    
    tests = [
        ("Dataset exists", test_dataset_exists),
        ("Dataset type supported", test_dataset_type),
        ("Dataset loading", test_dataset_loading),
        ("Embedding model", test_embedding_model)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- Testing: {name} ---")
        result = test_func()
        results.append(result)
        status = "PASSED" if result else "FAILED"
        print(f"--- {status}: {name} ---")
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)