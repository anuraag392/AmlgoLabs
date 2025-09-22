"""
Language Model Manager for loading and running inference with LLMs.
"""
import logging
import torch
from typing import Iterator, List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TextIteratorStreamer,
    GenerationConfig
)
from threading import Thread
import time

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages language model loading, configuration, and inference."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", 
                 max_context_length: int = 1024, temperature: float = 0.1,
                 max_new_tokens: int = 512):
        """
        Initialize LLMManager.
        
        Args:
            model_name: Name of the Hugging Face model
            max_context_length: Maximum context length for the model
            temperature: Sampling temperature for generation
            max_new_tokens: Maximum number of new tokens to generate
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"LLMManager initialized with model: {model_name}, device: {self.device}")
    
    def load_model(self):
        """Load the language model and tokenizer."""
        if self.model is not None:
            logger.info("Model already loaded")
            return
            
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                trust_remote_code=True
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_response(self, prompt: str, stream: bool = True) -> Iterator[str]:
        """
        Generate response from the model with optional streaming.
        
        Args:
            prompt: Input prompt for generation
            stream: Whether to stream the response
            
        Yields:
            Generated text tokens or complete response
        """
        if not prompt or not isinstance(prompt, str):
            logger.warning("Invalid prompt provided")
            return
            
        # Load model if not already loaded
        self.load_model()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length,
                padding=True
            ).to(self.device)
            
            # Create generation config
            generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
            
            if stream:
                # Streaming generation
                yield from self._generate_streaming(inputs, generation_config)
            else:
                # Non-streaming generation
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                yield response.strip()
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            yield f"Error generating response: {str(e)}"
    
    def _generate_streaming(self, inputs: Dict[str, torch.Tensor], 
                          generation_config: GenerationConfig) -> Iterator[str]:
        """
        Generate streaming response using TextIteratorStreamer.
        
        Args:
            inputs: Tokenized inputs
            generation_config: Generation configuration
            
        Yields:
            Generated text tokens
        """
        try:
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=30.0
            )
            
            # Update generation config with streamer
            generation_kwargs = {
                **inputs,
                **generation_config.to_dict(),
                'streamer': streamer
            }
            
            # Start generation in separate thread
            generation_thread = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            generation_thread.start()
            
            # Stream tokens
            generated_text = ""
            for token in streamer:
                if token:
                    generated_text += token
                    yield token
                    
            # Wait for generation to complete
            generation_thread.join()
            
            logger.info(f"Streaming generation completed: {len(generated_text)} characters")
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error in streaming generation: {str(e)}"
    
    def validate_response(self, response: str, context: List[str]) -> bool:
        """
        Validate that response is grounded in the provided context.
        
        Args:
            response: Generated response
            context: List of context chunks
            
        Returns:
            True if response appears to be grounded in context
        """
        if not response or not context:
            return False
            
        response_lower = response.lower()
        context_text = " ".join(context).lower()
        
        # Check for key phrases from context in response
        response_words = set(response_lower.split())
        context_words = set(context_text.split())
        
        # Calculate overlap
        overlap = len(response_words.intersection(context_words))
        overlap_ratio = overlap / len(response_words) if response_words else 0
        
        # Response should have reasonable overlap with context
        is_grounded = overlap_ratio > 0.3
        
        logger.info(f"Response validation: overlap_ratio={overlap_ratio:.3f}, grounded={is_grounded}")
        return is_grounded
    
    def create_prompt(self, query: str, context: List[str], 
                     system_message: Optional[str] = None) -> str:
        """
        Create a formatted prompt for the model.
        
        Args:
            query: User query
            context: List of relevant context chunks
            system_message: Optional system message
            
        Returns:
            Formatted prompt string
        """
        if not query:
            return ""
            
        # Default system message
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant. Answer the user's question based on the provided context. "
                "If the context doesn't contain enough information to answer the question, say so clearly. "
                "Do not make up information that is not in the context."
            )
        
        # Format context
        context_text = ""
        if context:
            context_text = "Context:\n"
            for i, chunk in enumerate(context, 1):
                context_text += f"{i}. {chunk}\n\n"
        
        # Create prompt
        prompt = f"{system_message}\n\n{context_text}Question: {query}\n\nAnswer:"
        
        # Ensure prompt doesn't exceed context length
        prompt = self._truncate_prompt(prompt)
        
        return prompt
    
    def _truncate_prompt(self, prompt: str) -> str:
        """
        Truncate prompt to fit within model's context length.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Truncated prompt
        """
        if self.tokenizer is None:
            # Rough estimation: 4 characters per token
            max_chars = self.max_context_length * 4
            if len(prompt) > max_chars:
                logger.warning(f"Truncating prompt from {len(prompt)} to {max_chars} characters")
                return prompt[:max_chars]
            return prompt
        
        # Tokenize to check length
        tokens = self.tokenizer.encode(prompt)
        
        if len(tokens) <= self.max_context_length:
            return prompt
        
        # Truncate by removing context chunks from the middle
        logger.warning(f"Prompt too long ({len(tokens)} tokens), truncating")
        
        # Find context section
        context_start = prompt.find("Context:\n")
        question_start = prompt.find("Question:")
        
        if context_start != -1 and question_start != -1:
            # Keep system message and question, truncate context
            system_part = prompt[:context_start]
            question_part = prompt[question_start:]
            
            # Calculate available space for context
            system_tokens = len(self.tokenizer.encode(system_part))
            question_tokens = len(self.tokenizer.encode(question_part))
            available_tokens = self.max_context_length - system_tokens - question_tokens - 50  # Buffer
            
            if available_tokens > 100:
                # Truncate context to fit
                context_part = prompt[context_start:question_start]
                context_tokens = self.tokenizer.encode(context_part)
                
                if len(context_tokens) > available_tokens:
                    truncated_context_tokens = context_tokens[:available_tokens]
                    truncated_context = self.tokenizer.decode(truncated_context_tokens)
                    prompt = system_part + truncated_context + "\n\n" + question_part
        
        return prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.model_name,
            'max_context_length': self.max_context_length,
            'temperature': self.temperature,
            'max_new_tokens': self.max_new_tokens,
            'device': self.device,
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None
        }
        
        if self.model is not None:
            info['model_parameters'] = sum(p.numel() for p in self.model.parameters())
            info['model_dtype'] = str(self.model.dtype)
        
        return info
    
    def update_generation_config(self, **kwargs):
        """
        Update generation configuration parameters.
        
        Args:
            **kwargs: Generation parameters to update
        """
        valid_params = ['temperature', 'max_new_tokens', 'max_context_length']
        
        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self, param, value)
                logger.info(f"Updated {param} to {value}")
            else:
                logger.warning(f"Invalid parameter: {param}")
    
    def clear_cache(self):
        """Clear model cache and free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared model cache")
    
    def unload_model(self):
        """Unload model from memory."""
        self.model = None
        self.tokenizer = None
        self.clear_cache()
        logger.info("Model unloaded from memory")