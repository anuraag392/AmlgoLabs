"""
Document processing module for cleaning and validating text documents.
"""
import re
import logging
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document cleaning, validation, and metadata extraction."""
    
    def __init__(self, min_word_count: int = 100):
        """
        Initialize DocumentProcessor.
        
        Args:
            min_word_count: Minimum word count for valid documents
        """
        self.min_word_count = min_word_count
        
    def clean_text(self, raw_text: str) -> str:
        """
        Clean and format raw text by removing unwanted elements.
        
        Args:
            raw_text: Raw input text
            
        Returns:
            Cleaned text string
        """
        if not raw_text or not isinstance(raw_text, str):
            return ""
            
        # Remove HTML tags if present
        text = self._remove_html(raw_text)
        
        # Remove headers and footers patterns
        text = self._remove_headers_footers(text)
        
        # Clean whitespace and formatting
        text = self._clean_whitespace(text)
        
        # Remove special characters and normalize
        text = self._normalize_text(text)
        
        logger.info(f"Cleaned text: {len(raw_text)} -> {len(text)} characters")
        return text
    
    def validate_document(self, text: str) -> bool:
        """
        Validate if document meets minimum requirements.
        
        Args:
            text: Text to validate
            
        Returns:
            True if document is valid, False otherwise
        """
        if not text or not isinstance(text, str):
            logger.warning("Document validation failed: empty or invalid text")
            return False
            
        # Check minimum word count
        word_count = len(text.split())
        if word_count < self.min_word_count:
            logger.warning(f"Document validation failed: {word_count} words < {self.min_word_count}")
            return False
            
        # Check for meaningful content (not just whitespace/punctuation)
        meaningful_chars = re.sub(r'[^\w\s]', '', text)
        if len(meaningful_chars.strip()) < self.min_word_count:
            logger.warning("Document validation failed: insufficient meaningful content")
            return False
            
        logger.info(f"Document validation passed: {word_count} words")
        return True
    
    def extract_metadata(self, document: str, source_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from document.
        
        Args:
            document: Document text
            source_path: Optional source file path
            
        Returns:
            Dictionary containing document metadata
        """
        word_count = len(document.split())
        char_count = len(document)
        
        # Extract potential title (first line or first sentence)
        lines = document.strip().split('\n')
        potential_title = lines[0].strip() if lines else ""
        if len(potential_title) > 100:
            # If first line is too long, use first sentence
            sentences = re.split(r'[.!?]+', document)
            potential_title = sentences[0].strip() if sentences else ""
            
        metadata = {
            "word_count": word_count,
            "char_count": char_count,
            "line_count": len(lines),
            "potential_title": potential_title[:100],  # Limit title length
            "source_path": source_path,
            "processed_at": datetime.now().isoformat(),
            "language": self._detect_language(document),
        }
        
        logger.info(f"Extracted metadata: {metadata}")
        return metadata
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text()
        return text
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header and footer patterns."""
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page\s+\d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove common footer patterns
        text = re.sub(r'\n\s*©.*?\n', '\n', text)
        text = re.sub(r'\n\s*Copyright.*?\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*All rights reserved.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove email signatures
        text = re.sub(r'\n\s*--\s*\n.*', '', text, flags=re.DOTALL)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text characters and encoding."""
        # Replace smart quotes and dashes
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '-', '…': '...'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Remove or replace other problematic characters
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', ' ', text)
        
        return text
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (English assumed for now)."""
        # Simple heuristic: if mostly ASCII and common English words, assume English
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text) if text else 0
        
        common_english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        english_word_count = sum(1 for word in common_english_words if word in text.lower())
        
        if ascii_ratio > 0.9 and english_word_count >= 3:
            return "en"
        else:
            return "unknown"