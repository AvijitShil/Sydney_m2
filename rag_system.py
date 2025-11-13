import os
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

# Use direct imports from available packages
try:
    # Try langchain_community imports first (newer versions)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
except ImportError:
    # Fall back to langchain imports (older versions)
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader

from config import CONFIG, MEDICAL_DATASET


class MedicalRAGSystem:
    """
    LangChain + FAISS RAG System with PDF Support
    - Chunks documents ONCE and saves to disk
    - Loads from disk on subsequent runs (instant)
    - Supports PDF, JSON, and TXT formats
    - Uses sentence-transformers embeddings
    """
    
    def __init__(self):
        print("\n" + "="*60)
        print("🔧 INITIALIZING LANGCHAIN RAG SYSTEM")
        print("="*60)
        
        # Configuration
        self.chunk_size = CONFIG.get("chunk_size", 500)
        self.chunk_overlap = CONFIG.get("chunk_overlap", 50)
        self.top_k = CONFIG.get("retrieval_k", 4)
        self.vector_db_path = CONFIG.get("vector_db_path", "./faiss_index")
        self.dataset_path = CONFIG.get("dataset_path")
        self.dataset_type = CONFIG.get("dataset_type", "pdf")
        self.similarity_score_threshold = CONFIG.get("similarity_score_threshold", 0.7)
        
        # Embedding model
        embedding_model_name = CONFIG.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"[+] Loading embedding model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("[✓] Embedding model loaded")
        
        # Vector store
        self.vectorstore = None
        self.query_cache = {}
        
        # Load or build index
        self._initialize_vectorstore()
        
    def _initialize_vectorstore(self):
        """Initialize the vector store - load from disk if exists, otherwise build from scratch."""
        if os.path.exists(self.vector_db_path) and os.path.isdir(self.vector_db_path):
            start_time = time.time()
            print(f"[+] Loading existing vector store from {self.vector_db_path}")
            try:
                # Try loading with compatibility mode
                self.vectorstore = FAISS.load_local(
                    self.vector_db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Check if index was loaded successfully
                load_time = time.time() - start_time
                print(f"[✓] Vector store loaded successfully in {load_time:.2f} seconds")
                
                # Try to get document count - handle both old and new formats
                try:
                    if hasattr(self.vectorstore, "_collection"):
                        doc_count = self.vectorstore._collection.count()
                        print(f"[i] Index contains {doc_count} documents")
                    elif hasattr(self.vectorstore, "index"):
                        doc_count = len(self.vectorstore.index_to_docstore_id)
                        print(f"[i] Index contains {doc_count} documents")
                    else:
                        print("[i] Unable to determine document count")
                except Exception:
                    print("[i] Unable to determine document count")
                    
            except Exception as e:
                print(f"[!] Error loading vector store: {e}")
                print("[+] Building new vector store...")
                self._build_vectorstore()
        else:
            print(f"[+] Vector store not found at {self.vector_db_path}")
            print("[+] Building new vector store...")
            self._build_vectorstore()
    
    def _build_vectorstore(self):
        """Build vector store from dataset."""
        start_time = time.time()
        
        # Load documents based on dataset type
        documents = self._load_documents()
        if not documents:
            print("[!] No documents loaded. Check dataset path and type.")
            return
        
        print(f"[+] Loaded {len(documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print(f"[+] Splitting documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        chunks = text_splitter.split_documents(documents)
        print(f"[+] Created {len(chunks)} chunks")
        
        # Create vector store
        print("[+] Creating FAISS index...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Save to disk
        print(f"[+] Saving vector store to {self.vector_db_path}")
        os.makedirs(self.vector_db_path, exist_ok=True)
        self.vectorstore.save_local(self.vector_db_path)
        
        build_time = time.time() - start_time
        print(f"[✓] Vector store built and saved in {build_time:.2f} seconds")
    
    def _load_documents(self) -> List[Document]:
        """Load documents from dataset."""
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            print(f"[!] Dataset path not found: {self.dataset_path}")
            return []
        
        print(f"[+] Loading dataset: {self.dataset_path}")
        
        try:
            if self.dataset_type.lower() == "pdf":
                loader = PyPDFLoader(self.dataset_path)
                return loader.load()
            elif self.dataset_type.lower() == "txt":
                loader = TextLoader(self.dataset_path)
                return loader.load()
            elif self.dataset_type.lower() == "json":
                # For JSON, we need to specify the jq schema to extract text
                loader = JSONLoader(
                    file_path=self.dataset_path,
                    jq_schema=".",  # Extract all content
                    text_content=False
                )
                return loader.load()
            else:
                print(f"[!] Unsupported dataset type: {self.dataset_type}")
                return []
        except Exception as e:
            print(f"[!] Error loading documents: {e}")
            return []
    
    def query(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant documents.
        
        Args:
            query_text: The query text
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        # Check cache first
        if query_text in self.query_cache:
            print(f"[i] Using cached results for query: {query_text}")
            return self.query_cache[query_text]
        
        print(f"[+] Querying: {query_text}")
        
        try:
            # Get similar documents with scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query_text, 
                k=self.top_k
            )
            
            # Filter by similarity threshold
            filtered_results = []
            for doc, score in docs_with_scores:
                # Convert cosine similarity to a 0-1 scale (higher is better)
                similarity = float(score)
                
                if similarity >= self.similarity_score_threshold:
                    result = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity": similarity
                    }
                    filtered_results.append(result)
            
            # Cache results
            self.query_cache[query_text] = filtered_results
            
            print(f"[✓] Found {len(filtered_results)} relevant documents")
            return filtered_results
            
        except Exception as e:
            print(f"[!] Error during query: {e}")
            return []
            
    def query_rag(self, query: str) -> List[str]:
        """Query the RAG system with a user question and get relevant context."""
        if not self.vectorstore:
            print("[!] Vector store not initialized")
            return []
        
        try:
            # Get relevant documents from the vector store
            docs = self.vectorstore.similarity_search_with_score(
                query, 
                k=self.top_k
            )
            
            # Filter by similarity score if threshold is set
            if self.similarity_score_threshold > 0:
                docs = [(doc, score) for doc, score in docs if score >= self.similarity_score_threshold]
            
            # Extract and return the content from documents
            results = []
            for doc, score in docs:
                results.append(doc.page_content)
                
            print(f"[✓] Found {len(results)} relevant chunks for query")
            return results
        except Exception as e:
            print(f"[!] RAG query error: {e}")
            # Try alternative approach if the error might be format-related
            try:
                print("[+] Attempting alternative query method...")
                # Use basic similarity search without scores as fallback
                docs = self.vectorstore.similarity_search(
                    query,
                    k=self.top_k
                )
                results = [doc.page_content for doc in docs]
                print(f"[✓] Found {len(results)} relevant chunks using fallback method")
                return results
            except Exception as e2:
                print(f"[!] Fallback query also failed: {e2}")
                return []
            
    def enhance_prompt_with_rag(self, query: str, system_prompt: str) -> str:
        """Enhance the system prompt with RAG context for the given query."""
        try:
            # Get relevant context from RAG
            context_chunks = self.query_rag(query)
            
            if not context_chunks:
                print("[i] No relevant context found for query")
                return system_prompt
                
            # Format the context to be included in the prompt
            context_text = "\n\n".join(context_chunks)
            
            # Enhance the system prompt with the context
            enhanced_prompt = f"{system_prompt}\n\nRELEVANT MEDICAL CONTEXT:\n{context_text}\n\nPlease use the above medical context to help answer the user's question accurately."
            
            return enhanced_prompt
        except Exception as e:
            print(f"[!] RAG enhancement error: {e}")
            return system_prompt
    
    def get_context_for_query(self, query_text: str) -> str:
        """
        Get formatted context for a query.
        
        Args:
            query_text: The query text
            
        Returns:
            Formatted context string
        """
        results = self.query(query_text)
        
        if not results:
            return "No relevant information found."
        
        # Format results
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result["content"].strip()
            metadata = result["metadata"]
            similarity = result["similarity"]
            
            # Format metadata
            meta_str = ""
            if "source" in metadata:
                meta_str += f"Source: {metadata['source']}"
            if "page" in metadata:
                meta_str += f", Page: {metadata['page']}"
                
            context_parts.append(f"[Document {i}] {content}\n{meta_str if meta_str else ''}")
        
        return "\n\n".join(context_parts)