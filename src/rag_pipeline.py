"""
RAG Pipeline module for coordinating retrieval and generation components.
"""
import logging
from typing import Iterator, List, Dict, Any, Optional
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from semantic_retriever import SemanticRetriever
from llm_manager import LLMManager
from prompt_template import PromptTemplate

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates the complete RAG workflow from query to response."""
    
    def __init__(self, retriever: SemanticRetriever, llm_manager: LLMManager,
                 prompt_template: PromptTemplate, max_context_chunks: int = 5,
                 enable_source_attribution: bool = True):
        """
        Initialize RAGPipeline.
        
        Args:
            retriever: Semantic retriever for finding relevant context
            llm_manager: Language model manager for response generation
            prompt_template: Template for formatting prompts
            max_context_chunks: Maximum number of context chunks to use
            enable_source_attribution: Whether to track and return source information
        """
        self.retriever = retriever
        self.llm_manager = llm_manager
        self.prompt_template = prompt_template
        self.max_context_chunks = max_context_chunks
        self.enable_source_attribution = enable_source_attribution
        
        # Pipeline statistics
        self.query_count = 0
        self.total_retrieval_time = 0.0
        self.total_generation_time = 0.0
        
        logger.info("RAGPipeline initialized with all components")
    
    def process_query(self, query: str, stream: bool = True, 
                     custom_system_message: Optional[str] = None,
                     retrieval_k: int = None) -> Iterator[Dict[str, Any]]:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: User query string
            stream: Whether to stream the response
            custom_system_message: Optional custom system message
            retrieval_k: Number of chunks to retrieve (overrides default)
            
        Yields:
            Dictionary containing response chunks and metadata
        """
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided to RAG pipeline")
            yield {
                'type': 'error',
                'content': 'Invalid query provided',
                'timestamp': datetime.now().isoformat()
            }
            return
        
        query = query.strip()
        if not query:
            logger.warning("Empty query after stripping")
            yield {
                'type': 'error',
                'content': 'Empty query provided',
                'timestamp': datetime.now().isoformat()
            }
            return
        
        self.query_count += 1
        pipeline_start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant context
            yield {
                'type': 'status',
                'content': 'Retrieving relevant context...',
                'timestamp': datetime.now().isoformat()
            }
            
            retrieval_start = datetime.now()
            k = retrieval_k if retrieval_k is not None else self.max_context_chunks
            context_chunks = self.retriever.retrieve_context(query, k=k)
            retrieval_time = (datetime.now() - retrieval_start).total_seconds()
            self.total_retrieval_time += retrieval_time
            
            logger.info(f"Retrieved {len(context_chunks)} context chunks in {retrieval_time:.2f}s")
            
            # Yield retrieval results
            yield {
                'type': 'retrieval_complete',
                'content': f'Found {len(context_chunks)} relevant chunks',
                'context_chunks': context_chunks,
                'retrieval_time': retrieval_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Step 2: Build prompt
            yield {
                'type': 'status',
                'content': 'Building prompt...',
                'timestamp': datetime.now().isoformat()
            }
            
            prompt = self.prompt_template.build_prompt(
                query=query,
                context_chunks=context_chunks,
                system_message=custom_system_message,
                include_sources=self.enable_source_attribution
            )
            
            # Validate prompt
            prompt_validation = self.prompt_template.validate_prompt_length(prompt)
            if not prompt_validation['is_valid']:
                logger.warning(f"Prompt length validation failed: {prompt_validation}")
            
            yield {
                'type': 'prompt_ready',
                'content': 'Prompt built successfully',
                'prompt_stats': prompt_validation,
                'timestamp': datetime.now().isoformat()
            }
            
            # Step 3: Generate response
            yield {
                'type': 'status',
                'content': 'Generating response...',
                'timestamp': datetime.now().isoformat()
            }
            
            generation_start = datetime.now()
            
            if stream:
                # Stream response generation
                response_text = ""
                for token in self.llm_manager.generate_response(prompt, stream=True):
                    if token:
                        response_text += token
                        yield {
                            'type': 'response_token',
                            'content': token,
                            'accumulated_response': response_text,
                            'timestamp': datetime.now().isoformat()
                        }
            else:
                # Non-streaming response
                response_generator = self.llm_manager.generate_response(prompt, stream=False)
                response_text = next(response_generator, "")
                
                yield {
                    'type': 'response_complete',
                    'content': response_text,
                    'timestamp': datetime.now().isoformat()
                }
            
            generation_time = (datetime.now() - generation_start).total_seconds()
            self.total_generation_time += generation_time
            
            # Step 4: Validate response and provide final results
            is_grounded = self._validate_response_grounding(response_text, context_chunks)
            
            pipeline_time = (datetime.now() - pipeline_start_time).total_seconds()
            
            # Final pipeline results
            yield {
                'type': 'pipeline_complete',
                'content': 'Pipeline processing complete',
                'final_response': response_text,
                'context_chunks': context_chunks if self.enable_source_attribution else [],
                'source_attribution': self._create_source_attribution(context_chunks) if self.enable_source_attribution else {},
                'validation': {
                    'is_grounded': is_grounded,
                    'context_chunks_used': len(context_chunks)
                },
                'performance': {
                    'total_time': pipeline_time,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'prompt_stats': prompt_validation
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            yield {
                'type': 'error',
                'content': f'Pipeline error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_response_grounding(self, response: str, context_chunks: List[Dict[str, Any]]) -> bool:
        """
        Validate that the response is grounded in the provided context.
        
        Args:
            response: Generated response
            context_chunks: Context chunks used for generation
            
        Returns:
            True if response appears grounded in context
        """
        if not response or not context_chunks:
            return False
        
        # Extract context text
        context_texts = []
        for chunk in context_chunks:
            text = chunk.get('metadata', {}).get('text', '')
            if text:
                context_texts.append(text)
        
        # Use LLM manager's validation
        return self.llm_manager.validate_response(response, context_texts)
    
    def _create_source_attribution(self, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create source attribution information from context chunks.
        
        Args:
            context_chunks: Context chunks with metadata
            
        Returns:
            Dictionary with source attribution information
        """
        sources = {}
        
        for i, chunk in enumerate(context_chunks):
            metadata = chunk.get('metadata', {})
            source_doc = metadata.get('source_document') or metadata.get('source_path', f'Unknown_{i}')
            
            if source_doc not in sources:
                sources[source_doc] = {
                    'chunks': [],
                    'avg_similarity': 0.0,
                    'chunk_count': 0
                }
            
            chunk_info = {
                'chunk_id': metadata.get('id', f'chunk_{i}'),
                'text_preview': metadata.get('text', '')[:100] + '...' if metadata.get('text', '') else '',
                'similarity_score': chunk.get('similarity_score', 0.0),
                'chunk_index': metadata.get('chunk_index', i)
            }
            
            sources[source_doc]['chunks'].append(chunk_info)
            sources[source_doc]['chunk_count'] += 1
        
        # Calculate average similarities
        for source_info in sources.values():
            if source_info['chunks']:
                similarities = [c['similarity_score'] for c in source_info['chunks']]
                source_info['avg_similarity'] = sum(similarities) / len(similarities)
        
        return {
            'sources': sources,
            'total_sources': len(sources),
            'total_chunks': len(context_chunks)
        }
    
    def process_query_batch(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch (non-streaming).
        
        Args:
            queries: List of query strings
            **kwargs: Additional arguments for process_query
            
        Returns:
            List of complete pipeline results
        """
        results = []
        
        for query in queries:
            logger.info(f"Processing batch query: {query[:50]}...")
            
            # Process query non-streaming
            pipeline_results = list(self.process_query(query, stream=False, **kwargs))
            
            # Extract final result
            final_result = None
            for result in pipeline_results:
                if result.get('type') == 'pipeline_complete':
                    final_result = result
                    break
            
            if final_result:
                results.append(final_result)
            else:
                # Handle error case
                error_result = {
                    'type': 'error',
                    'content': 'Failed to process query',
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
        
        return results
    
    async def process_query_async(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Process query asynchronously.
        
        Args:
            query: Query string
            **kwargs: Additional arguments for process_query
            
        Returns:
            List of pipeline results
        """
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor, 
                lambda: list(self.process_query(query, **kwargs))
            )
            return await future
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline performance statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        avg_retrieval_time = (
            self.total_retrieval_time / self.query_count 
            if self.query_count > 0 else 0.0
        )
        avg_generation_time = (
            self.total_generation_time / self.query_count 
            if self.query_count > 0 else 0.0
        )
        
        return {
            'query_count': self.query_count,
            'total_retrieval_time': self.total_retrieval_time,
            'total_generation_time': self.total_generation_time,
            'avg_retrieval_time': avg_retrieval_time,
            'avg_generation_time': avg_generation_time,
            'avg_total_time': avg_retrieval_time + avg_generation_time,
            'retriever_stats': self.retriever.get_retrieval_stats([]),
            'llm_info': self.llm_manager.get_model_info(),
            'template_info': self.prompt_template.get_template_info()
        }
    
    def update_configuration(self, **kwargs):
        """
        Update pipeline configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        if 'max_context_chunks' in kwargs:
            self.max_context_chunks = kwargs['max_context_chunks']
            logger.info(f"Updated max_context_chunks to {self.max_context_chunks}")
        
        if 'enable_source_attribution' in kwargs:
            self.enable_source_attribution = kwargs['enable_source_attribution']
            logger.info(f"Updated enable_source_attribution to {self.enable_source_attribution}")
        
        # Pass through configuration to components
        if 'similarity_threshold' in kwargs:
            self.retriever.update_similarity_threshold(kwargs['similarity_threshold'])
        
        if 'temperature' in kwargs or 'max_new_tokens' in kwargs:
            llm_params = {k: v for k, v in kwargs.items() 
                         if k in ['temperature', 'max_new_tokens', 'max_context_length']}
            if llm_params:
                self.llm_manager.update_generation_config(**llm_params)
        
        if 'template_name' in kwargs:
            self.prompt_template.set_template(kwargs['template_name'])
    
    def reset_stats(self):
        """Reset pipeline statistics."""
        self.query_count = 0
        self.total_retrieval_time = 0.0
        self.total_generation_time = 0.0
        logger.info("Pipeline statistics reset")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all pipeline components.
        
        Returns:
            Dictionary with health status of each component
        """
        health_status = {
            'pipeline': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # Check retriever
            retriever_stats = self.retriever.get_retrieval_stats([])
            health_status['components']['retriever'] = {
                'status': 'healthy',
                'config': {
                    'similarity_threshold': self.retriever.similarity_threshold,
                    'rerank_results': self.retriever.rerank_results
                }
            }
        except Exception as e:
            health_status['components']['retriever'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['pipeline'] = 'degraded'
        
        try:
            # Check LLM manager
            llm_info = self.llm_manager.get_model_info()
            health_status['components']['llm_manager'] = {
                'status': 'healthy',
                'model_loaded': llm_info['model_loaded'],
                'device': llm_info['device']
            }
        except Exception as e:
            health_status['components']['llm_manager'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['pipeline'] = 'degraded'
        
        try:
            # Check prompt template
            template_info = self.prompt_template.get_template_info()
            health_status['components']['prompt_template'] = {
                'status': 'healthy',
                'template_name': template_info['template_name'],
                'available_templates': len(template_info['available_templates'])
            }
        except Exception as e:
            health_status['components']['prompt_template'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['pipeline'] = 'degraded'
        
        return health_status