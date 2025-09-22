"""
Text chunking module for sentence-aware document splitting.
"""
import re
import logging
from typing import List, Dict, Any
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles sentence-aware text chunking with overlap."""
    
    def __init__(self, chunk_size: int = 250, overlap: int = 100, min_chunk_size: int = 100, max_chunk_size: int = 500):
        """
        Initialize TextChunker.
        
        Args:
            chunk_size: Target chunk size in words
            overlap: Number of words to overlap between chunks
            min_chunk_size: Minimum acceptable chunk size
            max_chunk_size: Maximum acceptable chunk size
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
            
        logger.info(f"TextChunker initialized: chunk_size={chunk_size}, overlap={overlap}")
    
    def sentence_aware_split(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK sentence tokenizer.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        if not text or not isinstance(text, str):
            return []
            
        # Use NLTK sentence tokenizer
        sentences = sent_tokenize(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # At least 3 words
                cleaned_sentences.append(sentence)
        
        logger.info(f"Split text into {len(cleaned_sentences)} sentences")
        return cleaned_sentences
    
    def create_chunks(self, text: str, source_document: str = "unknown") -> List[Dict[str, Any]]:
        """
        Create overlapping chunks from text using sentence-aware splitting.
        
        Args:
            text: Input text to chunk
            source_document: Source document identifier
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text or not isinstance(text, str):
            logger.warning("Cannot chunk empty or invalid text")
            return []
            
        sentences = self.sentence_aware_split(text)
        if not sentences:
            logger.warning("No valid sentences found in text")
            return []
            
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed max_chunk_size, finalize current chunk
            if current_word_count + sentence_words > self.max_chunk_size and current_chunk_sentences:
                chunk = self._create_chunk_dict(
                    current_chunk_sentences, 
                    chunk_index, 
                    source_document
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk_sentences)
                current_chunk_sentences = overlap_sentences
                current_word_count = sum(len(s.split()) for s in overlap_sentences)
                chunk_index += 1
            
            # Add current sentence
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_words
            
            # If we've reached target chunk size, try to finalize at sentence boundary
            if current_word_count >= self.chunk_size:
                # Look ahead to see if next sentence would make chunk too large
                if i + 1 < len(sentences):
                    next_sentence_words = len(sentences[i + 1].split())
                    if current_word_count + next_sentence_words > self.max_chunk_size:
                        # Finalize current chunk
                        chunk = self._create_chunk_dict(
                            current_chunk_sentences, 
                            chunk_index, 
                            source_document
                        )
                        chunks.append(chunk)
                        
                        # Start new chunk with overlap
                        overlap_sentences = self._get_overlap_sentences(current_chunk_sentences)
                        current_chunk_sentences = overlap_sentences
                        current_word_count = sum(len(s.split()) for s in overlap_sentences)
                        chunk_index += 1
        
        # Add final chunk if it has content
        if current_chunk_sentences:
            chunk = self._create_chunk_dict(
                current_chunk_sentences, 
                chunk_index, 
                source_document
            )
            chunks.append(chunk)
        
        # Validate and filter chunks
        valid_chunks = [chunk for chunk in chunks if self.validate_chunks([chunk])]
        
        logger.info(f"Created {len(valid_chunks)} valid chunks from {len(sentences)} sentences")
        return valid_chunks
    
    def validate_chunks(self, chunks: List[Dict]) -> bool:
        """
        Validate chunk quality and size requirements.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            True if all chunks are valid
        """
        if not chunks:
            return False
            
        for chunk in chunks:
            if not isinstance(chunk, dict):
                logger.warning("Invalid chunk format: not a dictionary")
                return False
                
            required_fields = ['id', 'text', 'word_count', 'chunk_index', 'source_document']
            for field in required_fields:
                if field not in chunk:
                    logger.warning(f"Invalid chunk: missing field '{field}'")
                    return False
            
            word_count = chunk['word_count']
            if word_count < self.min_chunk_size:
                logger.warning(f"Chunk too small: {word_count} words < {self.min_chunk_size}")
                return False
                
            if word_count > self.max_chunk_size:
                logger.warning(f"Chunk too large: {word_count} words > {self.max_chunk_size}")
                return False
                
            # Check text quality
            text = chunk['text']
            if not text or not isinstance(text, str):
                logger.warning("Invalid chunk text")
                return False
                
            # Check for meaningful content
            meaningful_chars = re.sub(r'[^\w\s]', '', text)
            if len(meaningful_chars.strip()) < 10:
                logger.warning("Chunk lacks meaningful content")
                return False
        
        return True
    
    def _create_chunk_dict(self, sentences: List[str], chunk_index: int, source_document: str) -> Dict[str, Any]:
        """
        Create a chunk dictionary from sentences.
        
        Args:
            sentences: List of sentences for the chunk
            chunk_index: Index of the chunk
            source_document: Source document identifier
            
        Returns:
            Chunk dictionary with metadata
        """
        text = ' '.join(sentences)
        word_count = len(text.split())
        
        chunk_id = f"{source_document}_chunk_{chunk_index:04d}"
        
        return {
            'id': chunk_id,
            'text': text,
            'word_count': word_count,
            'sentence_count': len(sentences),
            'chunk_index': chunk_index,
            'source_document': source_document,
            'created_at': datetime.now().isoformat(),
            'char_count': len(text),
        }
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """
        Get sentences for overlap with next chunk.
        
        Args:
            sentences: Current chunk sentences
            
        Returns:
            List of sentences for overlap
        """
        if not sentences:
            return []
            
        # Calculate how many words we want for overlap
        overlap_words = 0
        overlap_sentences = []
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if overlap_words + sentence_words <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_words += sentence_words
            else:
                break
        
        return overlap_sentences
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_words': 0,
                'avg_words_per_chunk': 0,
                'min_words': 0,
                'max_words': 0,
            }
        
        word_counts = [chunk['word_count'] for chunk in chunks]
        total_words = sum(word_counts)
        
        return {
            'total_chunks': len(chunks),
            'total_words': total_words,
            'avg_words_per_chunk': total_words / len(chunks),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'avg_sentences_per_chunk': sum(chunk.get('sentence_count', 0) for chunk in chunks) / len(chunks),
        }