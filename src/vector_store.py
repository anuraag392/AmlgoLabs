"""
Vector database module using FAISS for efficient similarity search.
"""
import logging
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector database for storing and searching embeddings."""
    
    def __init__(self, dimension: int, db_type: str = "faiss", index_type: str = "IndexFlatIP"):
        """
        Initialize VectorStore.
        
        Args:
            dimension: Dimension of the embeddings
            db_type: Type of vector database (currently only "faiss")
            index_type: FAISS index type ("IndexFlatIP" for cosine similarity)
        """
        self.dimension = dimension
        self.db_type = db_type
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self.is_trained = False
        
        if db_type != "faiss":
            raise ValueError(f"Unsupported database type: {db_type}")
            
        logger.info(f"VectorStore initialized: dimension={dimension}, index_type={index_type}")
    
    def create_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Create FAISS index from embeddings and metadata.
        
        Args:
            embeddings: NumPy array of embeddings with shape (n_samples, dimension)
            metadata: List of metadata dictionaries for each embedding
        """
        if embeddings.size == 0:
            raise ValueError("Cannot create index with empty embeddings")
            
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match expected {self.dimension}")
            
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(f"Metadata length {len(metadata)} doesn't match embeddings count {embeddings.shape[0]}")
        
        logger.info(f"Creating FAISS index with {embeddings.shape[0]} embeddings")
        
        try:
            # Create FAISS index based on type
            if self.index_type == "IndexFlatIP":
                # Inner Product index (good for normalized embeddings/cosine similarity)
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IndexFlatL2":
                # L2 distance index
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IndexIVFFlat":
                # IVF (Inverted File) index for faster search on large datasets
                quantizer = faiss.IndexFlatIP(self.dimension)
                nlist = min(100, max(1, embeddings.shape[0] // 10))  # Adaptive nlist
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            # Ensure embeddings are in the correct format
            embeddings = embeddings.astype(np.float32)
            
            # Train index if necessary (required for IVF indices)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                logger.info("Training FAISS index...")
                self.index.train(embeddings)
                self.is_trained = True
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            # Store metadata
            self.metadata = metadata.copy()
            
            logger.info(f"Successfully created FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add new embeddings to existing index.
        
        Args:
            embeddings: NumPy array of new embeddings
            metadata: List of metadata for new embeddings
        """
        if self.index is None:
            logger.info("No existing index, creating new one")
            self.create_index(embeddings, metadata)
            return
            
        if embeddings.size == 0:
            logger.warning("No embeddings to add")
            return
            
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match expected {self.dimension}")
            
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(f"Metadata length {len(metadata)} doesn't match embeddings count {embeddings.shape[0]}")
        
        try:
            embeddings = embeddings.astype(np.float32)
            
            # Add to index
            self.index.add(embeddings)
            
            # Add to metadata
            self.metadata.extend(metadata)
            
            logger.info(f"Added {embeddings.shape[0]} embeddings to index. Total: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to index: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            threshold: Optional similarity threshold for filtering results
            
        Returns:
            List of search results with metadata and similarity scores
        """
        if self.index is None:
            logger.warning("No index available for search")
            return []
            
        if query_embedding.size == 0:
            logger.warning("Empty query embedding")
            return []
            
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} doesn't match expected {self.dimension}")
        
        try:
            query_embedding = query_embedding.astype(np.float32)
            
            # Perform search
            scores, indices = self.index.search(query_embedding, k)
            
            # Process results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                    
                if threshold is not None and score < threshold:
                    continue
                    
                if idx >= len(self.metadata):
                    logger.warning(f"Index {idx} out of range for metadata")
                    continue
                
                result = {
                    'metadata': self.metadata[idx].copy(),
                    'similarity_score': float(score),
                    'rank': i + 1,
                    'index': int(idx)
                }
                results.append(result)
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def save_index(self, path: str):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            path: Base path for saving (without extension)
        """
        if self.index is None:
            raise ValueError("No index to save")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            index_path = f"{path}.faiss"
            faiss.write_index(self.index, index_path)
            
            # Save metadata and configuration
            metadata_path = f"{path}_metadata.pkl"
            save_data = {
                'metadata': self.metadata,
                'dimension': self.dimension,
                'db_type': self.db_type,
                'index_type': self.index_type,
                'is_trained': self.is_trained,
                'saved_at': datetime.now().isoformat(),
                'total_vectors': self.index.ntotal
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self, path: str):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            path: Base path for loading (without extension)
        """
        try:
            index_path = f"{path}.faiss"
            metadata_path = f"{path}_metadata.pkl"
            
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found: {index_path}")
                
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata and configuration
            with open(metadata_path, 'rb') as f:
                save_data = pickle.load(f)
            
            self.metadata = save_data['metadata']
            self.dimension = save_data['dimension']
            self.db_type = save_data['db_type']
            self.index_type = save_data['index_type']
            self.is_trained = save_data.get('is_trained', True)
            
            logger.info(f"Loaded index from {index_path} with {self.index.ntotal} vectors")
            logger.info(f"Index saved at: {save_data.get('saved_at', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        if self.index is None:
            return {
                'total_vectors': 0,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'is_trained': self.is_trained,
                'metadata_count': 0
            }
        
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'metadata_count': len(self.metadata)
        }
    
    def remove_vectors(self, indices: List[int]):
        """
        Remove vectors from the index (not supported by all FAISS indices).
        
        Args:
            indices: List of indices to remove
        """
        # Note: FAISS doesn't support removal for all index types
        # This is a placeholder for future implementation
        logger.warning("Vector removal not implemented for FAISS indices")
        raise NotImplementedError("Vector removal not supported by current FAISS index type")
    
    def get_vector_by_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a vector and its metadata by ID.
        
        Args:
            vector_id: ID of the vector to retrieve
            
        Returns:
            Dictionary with vector metadata or None if not found
        """
        for i, meta in enumerate(self.metadata):
            if meta.get('id') == vector_id:
                return {
                    'metadata': meta.copy(),
                    'index': i
                }
        
        return None
    
    def update_metadata(self, index: int, new_metadata: Dict[str, Any]):
        """
        Update metadata for a specific vector.
        
        Args:
            index: Index of the vector
            new_metadata: New metadata dictionary
        """
        if 0 <= index < len(self.metadata):
            self.metadata[index].update(new_metadata)
            logger.info(f"Updated metadata for vector at index {index}")
        else:
            raise IndexError(f"Index {index} out of range")
    
    def clear(self):
        """Clear the index and metadata."""
        self.index = None
        self.metadata = []
        self.is_trained = False
        logger.info("Cleared vector store")