"""
Response streaming module for real-time delivery with error recovery and buffering.
"""
import logging
import time
import asyncio
from typing import Iterator, Dict, Any, Optional, Callable, List
from datetime import datetime
from collections import deque
import threading
from queue import Queue, Empty
import json

from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class ResponseStreamer:
    """Handles streaming response delivery with buffering, error recovery, and performance optimization."""
    
    def __init__(self, pipeline: RAGPipeline, buffer_size: int = 100,
                 chunk_delay: float = 0.01, error_retry_attempts: int = 3,
                 fallback_to_non_streaming: bool = True):
        """
        Initialize ResponseStreamer.
        
        Args:
            pipeline: RAG pipeline for processing queries
            buffer_size: Size of the streaming buffer
            chunk_delay: Delay between chunks in seconds
            error_retry_attempts: Number of retry attempts on errors
            fallback_to_non_streaming: Whether to fallback to non-streaming on errors
        """
        self.pipeline = pipeline
        self.buffer_size = buffer_size
        self.chunk_delay = chunk_delay
        self.error_retry_attempts = error_retry_attempts
        self.fallback_to_non_streaming = fallback_to_non_streaming
        
        # Streaming state
        self.active_streams = {}
        self.stream_counter = 0
        self.performance_metrics = {
            'total_streams': 0,
            'successful_streams': 0,
            'failed_streams': 0,
            'fallback_streams': 0,
            'avg_stream_duration': 0.0,
            'total_tokens_streamed': 0
        }
        
        logger.info(f"ResponseStreamer initialized with buffer_size={buffer_size}")
    
    def stream_response(self, query: str, stream_id: Optional[str] = None,
                       on_token: Optional[Callable[[str], None]] = None,
                       on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
                       on_error: Optional[Callable[[Exception], None]] = None,
                       **pipeline_kwargs) -> Iterator[Dict[str, Any]]:
        """
        Stream response for a query with error recovery and buffering.
        
        Args:
            query: User query
            stream_id: Optional unique identifier for the stream
            on_token: Callback for each token
            on_complete: Callback when streaming completes
            on_error: Callback for errors
            **pipeline_kwargs: Additional arguments for pipeline
            
        Yields:
            Streaming response chunks with metadata
        """
        if stream_id is None:
            stream_id = f"stream_{self.stream_counter}"
            self.stream_counter += 1
        
        stream_start_time = time.time()
        self.performance_metrics['total_streams'] += 1
        
        # Initialize stream state
        stream_state = {
            'id': stream_id,
            'query': query,
            'start_time': stream_start_time,
            'buffer': deque(maxlen=self.buffer_size),
            'tokens_streamed': 0,
            'status': 'active',
            'error_count': 0
        }
        
        self.active_streams[stream_id] = stream_state
        
        try:
            logger.info(f"Starting stream {stream_id} for query: {query[:50]}...")
            
            # Attempt streaming with retry logic
            for attempt in range(self.error_retry_attempts + 1):
                try:
                    yield from self._stream_with_buffering(
                        stream_state, on_token, on_complete, **pipeline_kwargs
                    )
                    
                    # If we get here, streaming was successful
                    stream_state['status'] = 'completed'
                    self.performance_metrics['successful_streams'] += 1
                    break
                    
                except Exception as e:
                    stream_state['error_count'] += 1
                    logger.warning(f"Stream {stream_id} attempt {attempt + 1} failed: {e}")
                    
                    if attempt < self.error_retry_attempts:
                        # Wait before retry
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    else:
                        # All attempts failed
                        if self.fallback_to_non_streaming:
                            logger.info(f"Falling back to non-streaming for {stream_id}")
                            yield from self._fallback_non_streaming(
                                stream_state, on_complete, **pipeline_kwargs
                            )
                            self.performance_metrics['fallback_streams'] += 1
                        else:
                            # Re-raise the error
                            if on_error:
                                on_error(e)
                            raise
                        break
        
        except Exception as e:
            stream_state['status'] = 'failed'
            self.performance_metrics['failed_streams'] += 1
            logger.error(f"Stream {stream_id} failed completely: {e}")
            
            yield {
                'type': 'error',
                'stream_id': stream_id,
                'content': f'Streaming failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
            
            if on_error:
                on_error(e)
        
        finally:
            # Update performance metrics
            stream_duration = time.time() - stream_start_time
            self._update_performance_metrics(stream_state, stream_duration)
            
            # Clean up stream state
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    def _stream_with_buffering(self, stream_state: Dict[str, Any],
                              on_token: Optional[Callable[[str], None]],
                              on_complete: Optional[Callable[[Dict[str, Any]], None]],
                              **pipeline_kwargs) -> Iterator[Dict[str, Any]]:
        """
        Stream response with buffering and smooth delivery.
        
        Args:
            stream_state: Current stream state
            on_token: Token callback
            on_complete: Completion callback
            **pipeline_kwargs: Pipeline arguments
            
        Yields:
            Buffered streaming chunks
        """
        stream_id = stream_state['id']
        query = stream_state['query']
        buffer = stream_state['buffer']
        
        # Start pipeline processing
        pipeline_results = self.pipeline.process_query(query, stream=True, **pipeline_kwargs)
        
        accumulated_response = ""
        context_chunks = []
        final_result = None
        
        for result in pipeline_results:
            result_type = result.get('type')
            
            # Handle different result types
            if result_type == 'status':
                yield {
                    'type': 'stream_status',
                    'stream_id': stream_id,
                    'content': result['content'],
                    'timestamp': result['timestamp']
                }
            
            elif result_type == 'retrieval_complete':
                context_chunks = result.get('context_chunks', [])
                yield {
                    'type': 'stream_retrieval',
                    'stream_id': stream_id,
                    'content': result['content'],
                    'context_count': len(context_chunks),
                    'timestamp': result['timestamp']
                }
            
            elif result_type == 'response_token':
                token = result['content']
                accumulated_response = result.get('accumulated_response', accumulated_response + token)
                
                # Add to buffer
                buffer.append({
                    'token': token,
                    'timestamp': result['timestamp'],
                    'accumulated': accumulated_response
                })
                
                # Deliver buffered tokens with delay
                if len(buffer) >= self.buffer_size // 2 or token.endswith(('.', '!', '?', '\n')):
                    yield from self._flush_buffer(stream_state, on_token)
                
                stream_state['tokens_streamed'] += 1
                self.performance_metrics['total_tokens_streamed'] += 1
            
            elif result_type == 'pipeline_complete':
                final_result = result
                
                # Flush remaining buffer
                yield from self._flush_buffer(stream_state, on_token)
                
                # Yield final completion
                completion_data = {
                    'type': 'stream_complete',
                    'stream_id': stream_id,
                    'final_response': result['final_response'],
                    'context_chunks': result.get('context_chunks', []),
                    'source_attribution': result.get('source_attribution', {}),
                    'validation': result.get('validation', {}),
                    'performance': result.get('performance', {}),
                    'stream_stats': {
                        'tokens_streamed': stream_state['tokens_streamed'],
                        'buffer_flushes': getattr(stream_state, 'buffer_flushes', 0),
                        'duration': time.time() - stream_state['start_time']
                    },
                    'timestamp': result['timestamp']
                }
                
                yield completion_data
                
                if on_complete:
                    on_complete(completion_data)
            
            elif result_type == 'error':
                raise Exception(result['content'])
    
    def _flush_buffer(self, stream_state: Dict[str, Any],
                     on_token: Optional[Callable[[str], None]]) -> Iterator[Dict[str, Any]]:
        """
        Flush buffered tokens with smooth delivery.
        
        Args:
            stream_state: Current stream state
            on_token: Token callback
            
        Yields:
            Flushed token chunks
        """
        buffer = stream_state['buffer']
        stream_id = stream_state['id']
        
        while buffer:
            token_data = buffer.popleft()
            
            # Apply delivery delay for smooth streaming
            if self.chunk_delay > 0:
                time.sleep(self.chunk_delay)
            
            # Yield token
            yield {
                'type': 'stream_token',
                'stream_id': stream_id,
                'token': token_data['token'],
                'accumulated_response': token_data['accumulated'],
                'timestamp': token_data['timestamp']
            }
            
            # Call token callback
            if on_token:
                on_token(token_data['token'])
        
        # Track buffer flushes
        stream_state['buffer_flushes'] = getattr(stream_state, 'buffer_flushes', 0) + 1
    
    def _fallback_non_streaming(self, stream_state: Dict[str, Any],
                               on_complete: Optional[Callable[[Dict[str, Any]], None]],
                               **pipeline_kwargs) -> Iterator[Dict[str, Any]]:
        """
        Fallback to non-streaming response delivery.
        
        Args:
            stream_state: Current stream state
            on_complete: Completion callback
            **pipeline_kwargs: Pipeline arguments
            
        Yields:
            Non-streaming response chunks
        """
        stream_id = stream_state['id']
        query = stream_state['query']
        
        logger.info(f"Using non-streaming fallback for {stream_id}")
        
        yield {
            'type': 'stream_fallback',
            'stream_id': stream_id,
            'content': 'Switched to non-streaming mode due to errors',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Process query non-streaming
            pipeline_results = list(self.pipeline.process_query(query, stream=False, **pipeline_kwargs))
            
            # Find final result
            final_result = None
            for result in pipeline_results:
                if result.get('type') == 'pipeline_complete':
                    final_result = result
                    break
            
            if final_result:
                completion_data = {
                    'type': 'stream_complete',
                    'stream_id': stream_id,
                    'final_response': final_result['final_response'],
                    'context_chunks': final_result.get('context_chunks', []),
                    'source_attribution': final_result.get('source_attribution', {}),
                    'validation': final_result.get('validation', {}),
                    'performance': final_result.get('performance', {}),
                    'fallback_mode': True,
                    'timestamp': final_result['timestamp']
                }
                
                yield completion_data
                
                if on_complete:
                    on_complete(completion_data)
            else:
                raise Exception("No valid result from non-streaming fallback")
                
        except Exception as e:
            logger.error(f"Non-streaming fallback failed for {stream_id}: {e}")
            raise
    
    def _update_performance_metrics(self, stream_state: Dict[str, Any], duration: float):
        """
        Update performance metrics for completed stream.
        
        Args:
            stream_state: Completed stream state
            duration: Stream duration in seconds
        """
        # Update average duration
        total_streams = self.performance_metrics['total_streams']
        current_avg = self.performance_metrics['avg_stream_duration']
        
        new_avg = ((current_avg * (total_streams - 1)) + duration) / total_streams
        self.performance_metrics['avg_stream_duration'] = new_avg
        
        logger.info(f"Stream {stream_state['id']} completed in {duration:.2f}s, "
                   f"tokens: {stream_state['tokens_streamed']}")
    
    async def stream_response_async(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Asynchronous version of stream_response.
        
        Args:
            query: User query
            **kwargs: Additional arguments
            
        Returns:
            List of all streaming results
        """
        loop = asyncio.get_event_loop()
        
        # Run streaming in thread pool
        def run_streaming():
            return list(self.stream_response(query, **kwargs))
        
        return await loop.run_in_executor(None, run_streaming)
    
    def get_active_streams(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about currently active streams.
        
        Returns:
            Dictionary of active stream information
        """
        active_info = {}
        
        for stream_id, state in self.active_streams.items():
            active_info[stream_id] = {
                'query': state['query'][:50] + '...' if len(state['query']) > 50 else state['query'],
                'status': state['status'],
                'tokens_streamed': state['tokens_streamed'],
                'duration': time.time() - state['start_time'],
                'error_count': state['error_count']
            }
        
        return active_info
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get streaming performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        metrics = self.performance_metrics.copy()
        
        # Calculate success rate
        total = metrics['total_streams']
        if total > 0:
            metrics['success_rate'] = metrics['successful_streams'] / total
            metrics['failure_rate'] = metrics['failed_streams'] / total
            metrics['fallback_rate'] = metrics['fallback_streams'] / total
        else:
            metrics['success_rate'] = 0.0
            metrics['failure_rate'] = 0.0
            metrics['fallback_rate'] = 0.0
        
        # Add current active streams
        metrics['active_streams'] = len(self.active_streams)
        
        return metrics
    
    def update_config(self, **kwargs):
        """
        Update streamer configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        if 'buffer_size' in kwargs:
            self.buffer_size = kwargs['buffer_size']
            logger.info(f"Updated buffer_size to {self.buffer_size}")
        
        if 'chunk_delay' in kwargs:
            self.chunk_delay = kwargs['chunk_delay']
            logger.info(f"Updated chunk_delay to {self.chunk_delay}")
        
        if 'error_retry_attempts' in kwargs:
            self.error_retry_attempts = kwargs['error_retry_attempts']
            logger.info(f"Updated error_retry_attempts to {self.error_retry_attempts}")
        
        if 'fallback_to_non_streaming' in kwargs:
            self.fallback_to_non_streaming = kwargs['fallback_to_non_streaming']
            logger.info(f"Updated fallback_to_non_streaming to {self.fallback_to_non_streaming}")
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            'total_streams': 0,
            'successful_streams': 0,
            'failed_streams': 0,
            'fallback_streams': 0,
            'avg_stream_duration': 0.0,
            'total_tokens_streamed': 0
        }
        logger.info("Performance metrics reset")
    
    def stop_stream(self, stream_id: str) -> bool:
        """
        Stop an active stream.
        
        Args:
            stream_id: ID of stream to stop
            
        Returns:
            True if stream was stopped, False if not found
        """
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['status'] = 'stopped'
            logger.info(f"Stream {stream_id} stopped")
            return True
        
        return False
    
    def stop_all_streams(self):
        """Stop all active streams."""
        for stream_id in list(self.active_streams.keys()):
            self.stop_stream(stream_id)
        
        logger.info("All active streams stopped")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the streamer.
        
        Returns:
            Dictionary with health status
        """
        metrics = self.get_performance_metrics()
        
        # Determine health status
        if metrics['total_streams'] == 0:
            status = 'healthy'  # No streams yet
        elif metrics['success_rate'] > 0.8:
            status = 'healthy'
        elif metrics['success_rate'] > 0.5:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'active_streams': len(self.active_streams),
            'performance_metrics': metrics,
            'configuration': {
                'buffer_size': self.buffer_size,
                'chunk_delay': self.chunk_delay,
                'error_retry_attempts': self.error_retry_attempts,
                'fallback_to_non_streaming': self.fallback_to_non_streaming
            },
            'timestamp': datetime.now().isoformat()
        }
          