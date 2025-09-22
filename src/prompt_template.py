"""
Prompt template module for RAG-specific prompt generation and management.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Handles RAG-specific prompt templates with context injection and validation."""
    
    def __init__(self, template_name: str = "default_rag", max_context_chunks: int = 5,
                 max_prompt_length: int = 4000):
        """
        Initialize PromptTemplate.
        
        Args:
            template_name: Name of the template to use
            max_context_chunks: Maximum number of context chunks to include
            max_prompt_length: Maximum length of generated prompt in characters
        """
        self.template_name = template_name
        self.max_context_chunks = max_context_chunks
        self.max_prompt_length = max_prompt_length
        
        # Load template configurations
        self.templates = self._load_templates()
        
        logger.info(f"PromptTemplate initialized: {template_name}, max_chunks={max_context_chunks}")
    
    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]], 
                    system_message: Optional[str] = None,
                    include_sources: bool = True) -> str:
        """
        Build a complete RAG prompt with context injection.
        
        Args:
            query: User query
            context_chunks: List of retrieved context chunks with metadata
            system_message: Optional custom system message
            include_sources: Whether to include source information
            
        Returns:
            Formatted prompt string
        """
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided for prompt building")
            return ""
            
        query = query.strip()
        if not query:
            logger.warning("Empty query after stripping")
            return ""
        
        # Get template configuration
        template_config = self.templates.get(self.template_name, self.templates["default_rag"])
        
        # Use provided system message or template default
        if system_message is None:
            system_message = template_config["system_message"]
        
        # Limit context chunks
        limited_chunks = context_chunks[:self.max_context_chunks] if context_chunks else []
        
        # Format context section
        context_section = self._format_context_section(limited_chunks, include_sources)
        
        # Build prompt using template
        prompt_parts = []
        
        # Add system message
        if system_message:
            prompt_parts.append(system_message)
        
        # Add context if available
        if context_section:
            prompt_parts.append(context_section)
        
        # Add query section
        query_section = template_config["query_format"].format(query=query)
        prompt_parts.append(query_section)
        
        # Add answer prompt
        prompt_parts.append(template_config["answer_prompt"])
        
        # Join all parts
        prompt = template_config["section_separator"].join(prompt_parts)
        
        # Validate and truncate if necessary
        prompt = self._validate_and_truncate_prompt(prompt)
        
        logger.info(f"Built prompt: {len(prompt)} characters, {len(limited_chunks)} context chunks")
        return prompt
    
    def _format_context_section(self, context_chunks: List[Dict[str, Any]], 
                               include_sources: bool = True) -> str:
        """
        Format the context section of the prompt.
        
        Args:
            context_chunks: List of context chunks with metadata
            include_sources: Whether to include source information
            
        Returns:
            Formatted context section
        """
        if not context_chunks:
            return ""
        
        template_config = self.templates.get(self.template_name, self.templates["default_rag"])
        context_parts = [template_config["context_header"]]
        
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get('metadata', {})
            text = metadata.get('text', '')
            
            if not text:
                continue
            
            # Format individual context chunk
            chunk_text = f"{i}. {text}"
            
            # Add source information if requested
            if include_sources:
                source_info = self._format_source_info(metadata, chunk)
                if source_info:
                    chunk_text += f" {source_info}"
            
            context_parts.append(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def _format_source_info(self, metadata: Dict[str, Any], 
                           chunk: Dict[str, Any]) -> str:
        """
        Format source information for a context chunk.
        
        Args:
            metadata: Chunk metadata
            chunk: Full chunk information
            
        Returns:
            Formatted source information
        """
        source_parts = []
        
        # Add source document
        source_doc = metadata.get('source_document') or metadata.get('source_path')
        if source_doc:
            source_parts.append(f"Source: {source_doc}")
        
        # Add similarity score if available
        similarity = chunk.get('similarity_score')
        if similarity is not None:
            source_parts.append(f"Relevance: {similarity:.2f}")
        
        if source_parts:
            return f"({', '.join(source_parts)})"
        
        return ""
    
    def _validate_and_truncate_prompt(self, prompt: str) -> str:
        """
        Validate prompt and truncate if it exceeds maximum length.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Validated and potentially truncated prompt
        """
        if len(prompt) <= self.max_prompt_length:
            return prompt
        
        logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to {self.max_prompt_length}")
        
        # Try to truncate intelligently by removing context chunks from the end
        lines = prompt.split('\n')
        
        # Find context section
        context_start = -1
        context_end = -1
        
        for i, line in enumerate(lines):
            if "Context:" in line or "Relevant Information:" in line:
                context_start = i
            elif context_start != -1 and ("Question:" in line or "Query:" in line):
                context_end = i
                break
        
        if context_start != -1 and context_end != -1:
            # Remove context chunks from the end until we fit
            context_lines = lines[context_start + 1:context_end]
            
            while len('\n'.join(lines)) > self.max_prompt_length and context_lines:
                # Remove the last context chunk (find last numbered item)
                for i in range(len(context_lines) - 1, -1, -1):
                    if re.match(r'^\d+\.', context_lines[i].strip()):
                        # Remove this chunk and any following lines until next chunk or end
                        j = i + 1
                        while j < len(context_lines) and not re.match(r'^\d+\.', context_lines[j].strip()):
                            j += 1
                        context_lines = context_lines[:i] + context_lines[j:]
                        break
                else:
                    break
                
                # Rebuild lines
                lines = lines[:context_start + 1] + context_lines + lines[context_end:]
        
        # If still too long, do simple truncation
        truncated_prompt = '\n'.join(lines)
        if len(truncated_prompt) > self.max_prompt_length:
            truncated_prompt = truncated_prompt[:self.max_prompt_length - 3] + "..."
        
        return truncated_prompt
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Load prompt template configurations.
        
        Returns:
            Dictionary of template configurations
        """
        templates = {
            "default_rag": {
                "system_message": (
                    "You are a helpful AI assistant. Answer the user's question based on the provided context. "
                    "If the context doesn't contain enough information to answer the question, say so clearly. "
                    "Do not make up information that is not in the context. "
                    "Provide specific details and cite relevant information from the context when possible."
                ),
                "context_header": "Context:",
                "query_format": "Question: {query}",
                "answer_prompt": "Answer:",
                "section_separator": "\n\n"
            },
            
            "conversational_rag": {
                "system_message": (
                    "You are a knowledgeable and friendly AI assistant. Use the provided context to answer "
                    "the user's question in a conversational manner. If you cannot find the answer in the "
                    "context, politely explain what information is missing. Always be helpful and accurate."
                ),
                "context_header": "Relevant Information:",
                "query_format": "User Question: {query}",
                "answer_prompt": "Assistant Response:",
                "section_separator": "\n\n"
            },
            
            "technical_rag": {
                "system_message": (
                    "You are a technical expert AI assistant. Provide precise, detailed answers based on the "
                    "provided technical documentation. Include specific technical details, code examples, "
                    "and references when available in the context. If information is incomplete, specify "
                    "exactly what additional details would be needed."
                ),
                "context_header": "Technical Documentation:",
                "query_format": "Technical Query: {query}",
                "answer_prompt": "Technical Response:",
                "section_separator": "\n\n"
            },
            
            "summarization_rag": {
                "system_message": (
                    "You are an AI assistant specialized in summarization. Based on the provided context, "
                    "create a comprehensive summary that addresses the user's question. Focus on key points "
                    "and main ideas while maintaining accuracy to the source material."
                ),
                "context_header": "Source Material:",
                "query_format": "Summarization Request: {query}",
                "answer_prompt": "Summary:",
                "section_separator": "\n\n"
            }
        }
        
        return templates
    
    def get_available_templates(self) -> List[str]:
        """
        Get list of available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def set_template(self, template_name: str):
        """
        Set the active template.
        
        Args:
            template_name: Name of template to use
        """
        if template_name not in self.templates:
            available = ", ".join(self.get_available_templates())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")
        
        self.template_name = template_name
        logger.info(f"Switched to template: {template_name}")
    
    def add_custom_template(self, name: str, config: Dict[str, str]):
        """
        Add a custom template configuration.
        
        Args:
            name: Template name
            config: Template configuration dictionary
        """
        required_keys = ["system_message", "context_header", "query_format", 
                        "answer_prompt", "section_separator"]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Template config missing required key: {key}")
        
        self.templates[name] = config
        logger.info(f"Added custom template: {name}")
    
    def validate_prompt_length(self, prompt: str) -> Dict[str, Any]:
        """
        Validate prompt length and provide statistics.
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            Dictionary with validation results and statistics
        """
        char_count = len(prompt)
        word_count = len(prompt.split())
        line_count = len(prompt.split('\n'))
        
        # Estimate token count (rough approximation)
        estimated_tokens = char_count // 4
        
        is_valid = char_count <= self.max_prompt_length
        
        return {
            'is_valid': is_valid,
            'char_count': char_count,
            'word_count': word_count,
            'line_count': line_count,
            'estimated_tokens': estimated_tokens,
            'max_length': self.max_prompt_length,
            'length_ratio': char_count / self.max_prompt_length
        }
    
    def format_instruction_prompt(self, instruction: str, context_chunks: List[Dict[str, Any]],
                                 examples: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Format an instruction-based prompt for specific tasks.
        
        Args:
            instruction: Specific instruction for the task
            context_chunks: Relevant context chunks
            examples: Optional list of example input/output pairs
            
        Returns:
            Formatted instruction prompt
        """
        prompt_parts = []
        
        # Add instruction
        prompt_parts.append(f"Instruction: {instruction}")
        
        # Add examples if provided
        if examples:
            prompt_parts.append("Examples:")
            for i, example in enumerate(examples, 1):
                input_text = example.get('input', '')
                output_text = example.get('output', '')
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Input: {input_text}")
                prompt_parts.append(f"Output: {output_text}")
        
        # Add context
        if context_chunks:
            context_section = self._format_context_section(context_chunks, include_sources=False)
            prompt_parts.append(context_section)
        
        # Add input prompt
        prompt_parts.append("Input:")
        
        prompt = "\n\n".join(prompt_parts)
        return self._validate_and_truncate_prompt(prompt)
    
    def get_template_info(self) -> Dict[str, Any]:
        """
        Get information about the current template configuration.
        
        Returns:
            Dictionary with template information
        """
        current_config = self.templates.get(self.template_name, {})
        
        return {
            'template_name': self.template_name,
            'max_context_chunks': self.max_context_chunks,
            'max_prompt_length': self.max_prompt_length,
            'available_templates': self.get_available_templates(),
            'current_config': current_config,
            'system_message_length': len(current_config.get('system_message', '')),
        }
    
    def update_config(self, **kwargs):
        """
        Update template configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        valid_params = ['max_context_chunks', 'max_prompt_length']
        
        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self, param, value)
                logger.info(f"Updated {param} to {value}")
            else:
                logger.warning(f"Invalid configuration parameter: {param}")