"""
Semantic retrieval module for finding relevant document chunks.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from vector_store import VectorStore
from embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Handles semantic search and retrieval of relevant document chunks."""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingGenerator, 
                 similarity_threshold: float = 0.7, rerank_results: bool = True):
        """
        Initialize SemanticRetriever.
        
        Args:
            vector_store: Vector database for similarity search
            embedding_model: Model for generating query embeddings
            similarity_threshold: Minimum similarity score for results
            rerank_results: Whether to rerank results using additional criteria
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.rerank_results = rerank_results
        
        logger.info(f"SemanticRetriever initialized with threshold={similarity_threshold}")
    
    def retrieve_context(self, query: str, k: int = 5, 
                        min_similarity: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks for a query.
        
        Args:
            query: User query string
            k: Maximum number of results to return
            min_similarity: Override default similarity threshold
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided")
            return []
            
        query = query.strip()
        if not query:
            logger.warning("Empty query after stripping")
            return []
        
        logger.info(f"Retrieving context for query: '{query[:100]}...'")
        
        try:
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            if query_embedding.size == 0:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Perform similarity search
            threshold = min_similarity if min_similarity is not None else self.similarity_threshold
            raw_results = self.vector_store.search(
                query_embedding, 
                k=k * 2,  # Get more results for filtering and reranking
                threshold=threshold
            )
            
            if not raw_results:
                logger.info("No results found above similarity threshold")
                return []
            
            # Filter by relevance
            filtered_results = self.filter_relevance(raw_results, query, threshold)
            
            # Rerank results if enabled
            if self.rerank_results and len(filtered_results) > 1:
                reranked_results = self.rank_results(filtered_results, query)
            else:
                reranked_results = filtered_results
            
            # Limit to requested number of results
            final_results = reranked_results[:k]
            
            # Add retrieval metadata
            for i, result in enumerate(final_results):
                result['retrieval_rank'] = i + 1
                result['retrieved_at'] = datetime.now().isoformat()
                result['query'] = query
            
            logger.info(f"Retrieved {len(final_results)} relevant chunks")
            return final_results
            
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}")
            return []
    
    def rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Rerank search results using additional criteria.
        
        Args:
            results: List of search results
            query: Original query string
            
        Returns:
            Reranked list of results
        """
        if not results:
            return results
            
        logger.info(f"Reranking {len(results)} results")
        
        # Calculate additional ranking features
        for result in results:
            metadata = result.get('metadata', {})
            text = metadata.get('text', '')
            
            # Feature 1: Keyword overlap score
            keyword_score = self._calculate_keyword_overlap(query, text)
            
            # Feature 2: Text length preference (moderate length preferred)
            length_score = self._calculate_length_score(text)
            
            # Feature 3: Position in document (earlier chunks might be more important)
            position_score = self._calculate_position_score(metadata)
            
            # Feature 4: Semantic similarity (already available)
            similarity_score = result.get('similarity_score', 0.0)
            
            # Combine scores with weights
            combined_score = (
                0.4 * similarity_score +
                0.3 * keyword_score +
                0.2 * length_score +
                0.1 * position_score
            )
            
            result['rerank_score'] = combined_score
            result['keyword_score'] = keyword_score
            result['length_score'] = length_score
            result['position_score'] = position_score
        
        # Sort by combined score
        reranked = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        logger.info("Results reranked using combined scoring")
        return reranked
    
    def filter_relevance(self, results: List[Dict[str, Any]], query: str, 
                        threshold: float) -> List[Dict[str, Any]]:
        """
        Filter results based on relevance criteria.
        
        Args:
            results: List of search results
            query: Original query string
            threshold: Similarity threshold
            
        Returns:
            Filtered list of relevant results
        """
        if not results:
            return results
            
        filtered = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        for result in results:
            # Check similarity threshold
            similarity = result.get('similarity_score', 0.0)
            if similarity < threshold:
                continue
            
            metadata = result.get('metadata', {})
            text = metadata.get('text', '').lower()
            
            # Check for minimum keyword overlap
            text_words = set(re.findall(r'\b\w+\b', text))
            overlap = len(query_words.intersection(text_words))
            
            # Require at least one keyword match for very low similarity scores
            if similarity < threshold + 0.1 and overlap == 0:
                continue
            
            # Check text quality
            if not self._is_quality_text(text):
                continue
            
            filtered.append(result)
        
        logger.info(f"Filtered {len(results)} results to {len(filtered)} relevant ones")
        return filtered
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for the query."""
        try:
            embeddings = self.embedding_model.generate_embeddings([query])
            if embeddings.size > 0:
                return embeddings[0]
            else:
                return np.array([])
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return np.array([])
    
    def _calculate_keyword_overlap(self, query: str, text: str) -> float:
        """Calculate keyword overlap score between query and text."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        if not query_words:
            return 0.0
            
        overlap = len(query_words.intersection(text_words))
        return overlap / len(query_words)
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate score based on text length (prefer moderate length)."""
        word_count = len(text.split())
        
        # Optimal range: 100-300 words
        if 100 <= word_count <= 300:
            return 1.0
        elif word_count < 100:
            return word_count / 100.0
        else:
            # Penalize very long texts
            return max(0.5, 300.0 / word_count)
    
    def _calculate_position_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate score based on chunk position in document."""
        chunk_index = metadata.get('chunk_index', 0)
        
        # Prefer earlier chunks (they often contain key information)
        if chunk_index == 0:
            return 1.0
        elif chunk_index <= 5:
            return 0.8
        elif chunk_index <= 10:
            return 0.6
        else:
            return 0.4
    
    def _is_quality_text(self, text: str) -> bool:
        """Check if text meets quality criteria."""
        if not text or len(text.strip()) < 10:
            return False
            
        # Check for reasonable word count
        words = text.split()
        if len(words) < 5:
            return False
            
        # Check for too much repetition
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
            return False
            
        # Check for meaningful content (not just punctuation/numbers)
        meaningful_chars = re.sub(r'[^\w\s]', '', text)
        if len(meaningful_chars.strip()) < len(text) * 0.5:
            return False
            
        return True
    
    def get_retrieval_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about retrieval results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Dictionary with retrieval statistics
        """
        if not results:
            return {
                'total_results': 0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
                'avg_text_length': 0,
                'sources': []
            }
        
        similarities = [r.get('similarity_score', 0.0) for r in results]
        text_lengths = [len(r.get('metadata', {}).get('text', '').split()) for r in results]
        sources = list(set(r.get('metadata', {}).get('source_document', 'unknown') for r in results))
        
        return {
            'total_results': len(results),
            'avg_similarity': sum(similarities) / len(similarities),
            'min_similarity': min(similarities),
            'max_similarity': max(similarities),
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'sources': sources,
            'has_rerank_scores': any('rerank_score' in r for r in results)
        }
    
    def explain_retrieval(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate explanation of why these results were retrieved.
        
        Args:
            query: Original query
            results: Retrieval results
            
        Returns:
            Human-readable explanation
        """
        if not results:
            return f"No relevant results found for query: '{query}'"
        
        stats = self.get_retrieval_stats(results)
        
        explanation = f"Found {stats['total_results']} relevant chunks for query: '{query}'\n\n"
        explanation += f"Average similarity: {stats['avg_similarity']:.3f}\n"
        explanation += f"Similarity range: {stats['min_similarity']:.3f} - {stats['max_similarity']:.3f}\n"
        explanation += f"Average text length: {stats['avg_text_length']:.0f} words\n"
        explanation += f"Sources: {', '.join(stats['sources'])}\n"
        
        if stats['has_rerank_scores']:
            explanation += "\nResults were reranked using combined scoring (similarity + keywords + length + position)\n"
        
        explanation += "\nTop results:\n"
        for i, result in enumerate(results[:3], 1):
            metadata = result.get('metadata', {})
            similarity = result.get('similarity_score', 0.0)
            text_preview = metadata.get('text', '')[:100] + "..."
            
            explanation += f"{i}. Similarity: {similarity:.3f} - {text_preview}\n"
        
        return explanation
    
    def update_similarity_threshold(self, new_threshold: float):
        """Update the similarity threshold."""
        if 0.0 <= new_threshold <= 1.0:
            self.similarity_threshold = new_threshold
            logger.info(f"Updated similarity threshold to {new_threshold}")
        else:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
    
    def toggle_reranking(self, enable: bool):
        """Enable or disable result reranking."""
        self.rerank_results = enable
        logger.info(f"Result reranking {'enabled' if enable else 'disabled'}")