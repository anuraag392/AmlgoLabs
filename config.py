"""
Configuration settings for the RAG Chatbot application.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    """Configuration class for RAG Chatbot system."""
    
    # Document Processing
    chunk_size: int = 200
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 300
    
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    batch_size: int = 32
    
    # Vector Database
    vector_db_type: str = "faiss"
    similarity_threshold: float = 0.7
    index_type: str = "IndexFlatIP"  # Inner Product for cosine similarity
    
    # LLM
    llm_model: str = "microsoft/DialoGPT-medium"  # Smaller model for demo
    max_context_length: int = 1024
    temperature: float = 0.1
    max_new_tokens: int = 512
    do_sample: bool = True
    
    # Retrieval
    top_k_chunks: int = 5
    rerank_results: bool = True
    
    # Streaming
    stream_chunk_size: int = 1
    stream_delay: float = 0.01
    
    # Paths
    data_dir: str = "data"
    chunks_dir: str = "chunks"
    vectordb_dir: str = "vectordb"
    models_dir: str = "models"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/rag_chatbot.log"
    
    # UI
    page_title: str = "RAG Chatbot - Amlgo Labs"
    page_icon: str = "🤖"
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables."""
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", 200)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_model=os.getenv("LLM_MODEL", "microsoft/DialoGPT-medium"),
            temperature=float(os.getenv("TEMPERATURE", 0.1)),
            top_k_chunks=int(os.getenv("TOP_K_CHUNKS", 5)),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# Global configuration instance
config = RAGConfig.from_env()