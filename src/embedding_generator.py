"""
Embedding generation module for creating semantic embeddings from text.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import pickle
import os

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles semantic embedding generation using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32, cache_dir: Optional[str] = None):
        """
        Initialize EmbeddingGenerator.
        
        Args:
            model_name: Name of the sentence transformer model
            batch_size: Batch size for processing
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.model = None
        self.embedding_cache = {}
        
        # Create cache directory if specified
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")
    
    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                
                # Check if CUDA is available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(device)
                logger.info(f"Model loaded on device: {device}")
                
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return np.array([])
            
        # Filter out empty texts
        valid_texts = [text for text in texts if text and isinstance(text, str) and text.strip()]
        if not valid_texts:
            logger.warning("No valid texts found after filtering")
            return np.array([])
            
        # Load model if not already loaded
        self._load_model()
        
        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            
            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(valid_texts), self.batch_size):
                batch_texts = valid_texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True  # Normalize for cosine similarity
                )
                all_embeddings.append(batch_embeddings)
                
                if i % (self.batch_size * 10) == 0:  # Log progress every 10 batches
                    logger.info(f"Processed {min(i + self.batch_size, len(valid_texts))}/{len(valid_texts)} texts")
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def batch_process(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks in batches to add embeddings.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            logger.warning("No chunks provided for batch processing")
            return []
            
        logger.info(f"Batch processing {len(chunks)} chunks")
        
        # Extract texts from chunks
        texts = []
        valid_chunk_indices = []
        
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict) and 'text' in chunk:
                text = chunk['text']
                if text and isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                    valid_chunk_indices.append(i)
                else:
                    logger.warning(f"Skipping chunk {i} with invalid text")
            else:
                logger.warning(f"Skipping invalid chunk {i}")
        
        if not texts:
            logger.warning("No valid texts found in chunks")
            return chunks
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        if embeddings.size == 0:
            logger.warning("No embeddings generated")
            return chunks
        
        # Add embeddings to chunks
        processed_chunks = chunks.copy()
        for i, chunk_idx in enumerate(valid_chunk_indices):
            processed_chunks[chunk_idx] = processed_chunks[chunk_idx].copy()
            processed_chunks[chunk_idx]['embedding'] = embeddings[i]
            processed_chunks[chunk_idx]['embedding_model'] = self.model_name
            processed_chunks[chunk_idx]['embedding_generated_at'] = datetime.now().isoformat()
        
        logger.info(f"Successfully added embeddings to {len(valid_chunk_indices)} chunks")
        return processed_chunks
    
    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        Validate embedding array quality.
        
        Args:
            embeddings: NumPy array of embeddings
            
        Returns:
            True if embeddings are valid
        """
        if embeddings is None or embeddings.size == 0:
            logger.warning("Embeddings are empty or None")
            return False
            
        if not isinstance(embeddings, np.ndarray):
            logger.warning("Embeddings are not a NumPy array")
            return False
            
        if len(embeddings.shape) != 2:
            logger.warning(f"Embeddings have wrong shape: {embeddings.shape}, expected 2D")
            return False
            
        # Check for NaN or infinite values
        if np.isnan(embeddings).any():
            logger.warning("Embeddings contain NaN values")
            return False
            
        if np.isinf(embeddings).any():
            logger.warning("Embeddings contain infinite values")
            return False
            
        # Check embedding dimension is reasonable
        embedding_dim = embeddings.shape[1]
        if embedding_dim < 50 or embedding_dim > 2048:
            logger.warning(f"Unusual embedding dimension: {embedding_dim}")
            return False
            
        # Check if embeddings are normalized (for cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            logger.info("Embeddings are not normalized, normalizing now")
            embeddings = embeddings / norms[:, np.newaxis]
        
        logger.info(f"Embeddings validation passed: shape {embeddings.shape}")
        return True
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Normalized embeddings
        """
        if embeddings.size == 0:
            return embeddings
            
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        
        logger.info("Embeddings normalized for cosine similarity")
        return normalized
    
    def save_embeddings_cache(self, cache_key: str, embeddings: np.ndarray):
        """
        Save embeddings to cache.
        
        Args:
            cache_key: Unique key for the embeddings
            embeddings: Embeddings to cache
        """
        if not self.cache_dir:
            return
            
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save embeddings cache: {e}")
    
    def load_embeddings_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Load embeddings from cache.
        
        Args:
            cache_key: Unique key for the embeddings
            
        Returns:
            Cached embeddings or None if not found
        """
        if not self.cache_dir:
            return None
            
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings cache: {cache_path}")
                return embeddings
        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}")
            
        return None
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            Embedding dimension
        """
        self._load_model()
        
        # Generate a test embedding to get dimension
        test_embedding = self.model.encode(["test"], convert_to_numpy=True)
        return test_embedding.shape[1]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
            
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        self._load_model()
        
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension(),
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'device': str(self.model.device) if self.model else 'unknown',
            'batch_size': self.batch_size,
        }