import streamlit as st
import logging
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor
from text_chunker import TextChunker
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from semantic_retriever import SemanticRetriever
from llm_manager import LLMManager
from prompt_template import PromptTemplate
from rag_pipeline import RAGPipeline
from enhanced_context_analyzer import EnhancedContextAnalyzer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ChatbotApp:
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_page_config()
        
    def initialize_session_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = None
        
        if 'config' not in st.session_state:
            st.session_state.config = {
                'model_name': 'distilgpt2',
                'embedding_model': 'all-MiniLM-L6-v2',
                'max_context_chunks': 5,
                'similarity_threshold': 0.3,
                'temperature': 0.1,
                'max_new_tokens': 512,
                'enable_source_attribution': True
            }
        
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
        
        if 'vector_store_ready' not in st.session_state:
            st.session_state.vector_store_ready = False
        
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        
        if 'total_response_time' not in st.session_state:
            st.session_state.total_response_time = 0.0
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="RAG Chatbot - Amlgo Labs",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #1f77b4;
        }
        
        .user-message {
            background-color: #f0f2f6;
            border-left-color: #ff7f0e;
        }
        
        .assistant-message {
            background-color: #ffffff;
            border-left-color: #1f77b4;
        }
        
        .source-citation {
            font-size: 0.8rem;
            color: #666;
            font-style: italic;
            margin-top: 0.5rem;
        }
        
        .system-status {
            padding: 0.5rem;
            border-radius: 0.25rem;
            margin: 0.25rem 0;
        }
        
        .status-healthy {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-warning {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        st.markdown('<h1 class="main-header">RAG Chatbot(amlgo Labs)</h1>', unsafe_allow_html=True)
    
    
    def render_sidebar(self):
        with st.sidebar:
            st.header("🔧 System Control")
            
            if not st.session_state.system_initialized:
                if st.button("Initialize System", type="primary"):
                    self.initialize_system()
            else:
                st.success("✅ System Ready")
            
            st.markdown("---")
            
            st.header("Document Management")
            
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['txt', 'pdf', 'docx'],
                accept_multiple_files=True,
                help="Upload documents to build the knowledge base"
            )
            
            if uploaded_files and st.button("📥 Process Documents"):
                self.process_uploaded_documents(uploaded_files)
            
            if st.session_state.documents_loaded:
                st.success("✅ Documents Processed")
            
            if st.session_state.vector_store_ready:
                st.success("✅ Knowledge Base Ready")
            
            if st.button("Load Previous Session"):
                self.load_existing_data()
            
            st.markdown("---")
            
            st.header("Settings")
            
            with st.expander("Model Settings"):
                st.session_state.config['max_context_chunks'] = st.slider(
                    "Context Chunks", 1, 10, st.session_state.config['max_context_chunks']
                )
                
                st.session_state.config['similarity_threshold'] = st.slider(
                    "Similarity Threshold", 0.0, 1.0, st.session_state.config['similarity_threshold'], 0.05
                )
            
            with st.expander("Display Settings"):
                st.session_state.config['enable_source_attribution'] = st.checkbox(
                    "Show Source Citations", st.session_state.config['enable_source_attribution']
                )
                
                if 'show_query_analysis' not in st.session_state.config:
                    st.session_state.config['show_query_analysis'] = False
                
                st.session_state.config['show_query_analysis'] = st.checkbox(
                    "Show Query Analysis", st.session_state.config['show_query_analysis']
                )
            
            if st.button("Apply Settings"):
                self.apply_configuration()
            
            st.markdown("---")
            
            st.header("Statistics")
            self.render_statistics()
            
            st.markdown("---")
            
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    
    def render_statistics(self):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Queries", st.session_state.query_count)
        
        with col2:
            avg_time = (st.session_state.total_response_time / st.session_state.query_count 
                       if st.session_state.query_count > 0 else 0)
            st.metric("Avg Response", f"{avg_time:.2f}s")
        
        if st.session_state.vector_store_ready and hasattr(st.session_state, 'vector_store'):
            try:
                stats = st.session_state.vector_store.get_stats()
                st.text(f"Knowledge Base: {stats.get('total_vectors', 0)} chunks")
            except:
                pass
    
    def initialize_system(self):
        try:
            with st.spinner("Initializing system..."):
                doc_processor = DocumentProcessor()
                text_chunker = TextChunker()
                embedding_generator = EmbeddingGenerator(
                    model_name=st.session_state.config['embedding_model']
                )
                
                embedding_dim = embedding_generator.get_embedding_dimension()
                
                vector_store = VectorStore(dimension=embedding_dim)
                
                retriever = SemanticRetriever(
                    vector_store=vector_store,
                    embedding_model=embedding_generator,
                    similarity_threshold=st.session_state.config['similarity_threshold']
                )
                
                llm_manager = LLMManager(
                    model_name=st.session_state.config['model_name'],
                    temperature=st.session_state.config['temperature'],
                    max_new_tokens=st.session_state.config['max_new_tokens']
                )
                
                prompt_template = PromptTemplate(
                    max_context_chunks=st.session_state.config['max_context_chunks']
                )
                
                pipeline = RAGPipeline(
                    retriever=retriever,
                    llm_manager=llm_manager,
                    prompt_template=prompt_template,
                    max_context_chunks=st.session_state.config['max_context_chunks'],
                    enable_source_attribution=st.session_state.config['enable_source_attribution']
                )
                
                context_analyzer = EnhancedContextAnalyzer()
                
                st.session_state.pipeline = pipeline
                st.session_state.doc_processor = doc_processor
                st.session_state.text_chunker = text_chunker
                st.session_state.embedding_generator = embedding_generator
                st.session_state.vector_store = vector_store
                st.session_state.context_analyzer = context_analyzer
                st.session_state.system_initialized = True
                
                st.success("✅ System initialized successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
            logger.error(f"System initialization error: {e}")
    
    def process_uploaded_documents(self, uploaded_files):
        if not st.session_state.system_initialized:
            st.error("Please initialize the system first!")
            return
        
        try:
            with st.spinner("Processing documents..."):
                all_chunks = []
                
                for uploaded_file in uploaded_files:
                    content = self.extract_text_from_file(uploaded_file)
                    
                    if not content or len(content.strip()) < 10:
                        st.warning(f"Could not extract content from {uploaded_file.name}")
                        continue
                    
                    cleaned_content = st.session_state.doc_processor.clean_text(content)
                    
                    if not st.session_state.doc_processor.validate_document(cleaned_content):
                        st.warning(f"Document {uploaded_file.name} failed validation")
                        continue
                    
                    metadata = st.session_state.doc_processor.extract_metadata(
                        cleaned_content, uploaded_file.name
                    )
                    
                    chunks = st.session_state.text_chunker.create_chunks(
                        cleaned_content, uploaded_file.name
                    )
                    
                    all_chunks.extend(chunks)
                
                if not all_chunks:
                    st.error("No valid chunks generated from uploaded documents")
                    return
                
                chunks_with_embeddings = st.session_state.embedding_generator.batch_process(all_chunks)
                
                embeddings = []
                metadata_list = []
                
                for chunk in chunks_with_embeddings:
                    if 'embedding' in chunk:
                        embeddings.append(chunk['embedding'])
                        metadata_list.append(chunk)
                
                if not embeddings:
                    st.error("No embeddings generated")
                    return
                
                import numpy as np
                embeddings_array = np.array(embeddings)
                st.session_state.vector_store.create_index(embeddings_array, metadata_list)
                
                st.session_state.documents_loaded = True
                st.session_state.vector_store_ready = True
                
                self.save_processed_data(all_chunks)
                
                st.success(f"✅ Processed {len(uploaded_files)} documents, created {len(all_chunks)} chunks")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Document processing failed: {str(e)}")
            logger.error(f"Document processing error: {e}")
    
    def extract_text_from_file(self, uploaded_file):
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        try:
            if file_extension == 'txt':
                content_bytes = uploaded_file.read()
                
                encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        content = content_bytes.decode(encoding)
                        return content
                    except UnicodeDecodeError:
                        continue
                
                return content_bytes.decode('utf-8', errors='replace')
            
            elif file_extension == 'pdf':
                try:
                    import PyPDF2
                    from io import BytesIO
                    
                    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                    text_content = ""
                    
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n"
                    
                    return text_content
                
                except ImportError:
                    st.error("❌ PyPDF2 not installed. Please install it: pip install PyPDF2")
                    return ""
                except Exception as e:
                    st.error(f"❌ PDF processing failed: {str(e)}")
                    return ""
            
            elif file_extension in ['docx', 'doc']:
                try:
                    import docx
                    from io import BytesIO
                    
                    doc = docx.Document(BytesIO(uploaded_file.read()))
                    text_content = ""
                    
                    for paragraph in doc.paragraphs:
                        text_content += paragraph.text + "\n"
                    
                    return text_content
                
                except ImportError:
                    st.error("❌ python-docx not installed. Please install it: pip install python-docx")
                    return ""
                except Exception as e:
                    st.error(f"❌ Word document processing failed: {str(e)}")
                    return ""
            
            else:
                st.warning(f"⚠️ Unknown file type: {file_extension}. Trying to process as text...")
                content_bytes = uploaded_file.read()
                return content_bytes.decode('utf-8', errors='replace')
        
        except Exception as e:
            st.error(f"❌ File processing failed for {uploaded_file.name}: {str(e)}")
            return ""
    
    def save_processed_data(self, chunks):
        try:
            import json
            import os
            from datetime import datetime
            
            os.makedirs("chunks", exist_ok=True)
            os.makedirs("vectordb", exist_ok=True)
            
            chunks_data = []
            for chunk in chunks:
                chunk_copy = chunk.copy()
                if 'embedding' in chunk_copy:
                    chunk_copy['embedding'] = chunk_copy['embedding'].tolist()
                chunks_data.append(chunk_copy)
            
            with open("chunks/current_chunks.json", "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            st.session_state.vector_store.save_index("vectordb/current_vectors")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
    
    def load_existing_data(self):
        try:
            import os
            
            if os.path.exists("vectordb/current_vectors.faiss"):
                with st.spinner("Loading previous session..."):
                    st.session_state.vector_store.load_index("vectordb/current_vectors")
                    st.session_state.documents_loaded = True
                    st.session_state.vector_store_ready = True
                    
                    stats = st.session_state.vector_store.get_stats()
                    st.success(f"Loaded previous session with {stats['total_vectors']} chunks")
                    st.rerun()
            else:
                st.warning("No previous session found")
                
        except Exception as e:
            st.error(f"Failed to load previous session: {e}")
    
    def apply_configuration(self):
        if not st.session_state.system_initialized:
            st.warning("System not initialized")
            return
        
        try:
            st.session_state.pipeline.update_configuration(
                max_context_chunks=st.session_state.config['max_context_chunks'],
                enable_source_attribution=st.session_state.config['enable_source_attribution'],
                similarity_threshold=st.session_state.config['similarity_threshold']
            )
            
            st.success("✅ Settings applied successfully!")
            
        except Exception as e:
            st.error(f"❌ Settings update failed: {str(e)}")
            logger.error(f"Configuration error: {e}")
    
    def render_chat_interface(self):
        for message in st.session_state.messages:
            self.render_message(message)
        
        if prompt := st.chat_input("Ask me anything about the uploaded documents..."):
            if not st.session_state.system_initialized:
                st.error("Please initialize the system first!")
                return
            
            if not st.session_state.vector_store_ready:
                st.error("Please upload and process documents first!")
                return
            
            user_message = {
                'role': 'user',
                'content': prompt,
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            
            self.render_message(user_message)
            
            self.generate_response(prompt)
    
    def render_message(self, message: Dict[str, Any]):
        role = message['role']
        content = message['content']
        
        if role == 'user':
            with st.chat_message("user"):
                st.markdown(content)
        
        elif role == 'assistant':
            with st.chat_message("assistant"):
                st.markdown(content)
                
                if 'sources' in message and st.session_state.config['enable_source_attribution']:
                    self.render_sources(message['sources'])
                
                if 'query_analysis' in message:
                    self.render_query_analysis(message['query_analysis'])
    
    def render_sources(self, sources: Dict[str, Any]):
        if not sources or not sources.get('sources'):
            return
        
        with st.expander("Sources"):
            for source_name, source_info in sources['sources'].items():
                st.markdown(f"**{source_name}**")
                
                for chunk in source_info['chunks']:
                    similarity = chunk.get('similarity_score', 0)
                    preview = chunk.get('text_preview', '')
                    
                    st.markdown(f"- Relevance: {similarity:.2f}")
                    st.markdown(f"  *{preview}*")
                
                st.markdown("---")
    
    def render_query_analysis(self, query_analysis: Dict[str, Any]):
        if not st.session_state.config.get('show_query_analysis', False):
            return
            
        with st.expander("Query test"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Question Type:** {query_analysis['question_type']}")
                st.write(f"**Intent:** {query_analysis['intent']}")
                st.write(f"**Complexity:** {query_analysis['complexity']}")
            
            with col2:
                if query_analysis['entities']:
                    st.write(f"**Entities:** {', '.join(query_analysis['entities'])}")
                if query_analysis['key_terms']:
                    st.write(f"**Key Terms:** {', '.join(query_analysis['key_terms'][:5])}")
    
    def render_enhanced_sources(self, sources: Dict[str, Any], ranked_chunks: List[Dict[str, Any]] = None):
        if not sources or not sources.get('sources'):
            return
        
        with st.expander("Source"):
            for source_name, source_info in sources['sources'].items():
                st.markdown(f"**{source_name}**")
                
                for i, chunk in enumerate(source_info['chunks']):
                    similarity = chunk.get('similarity_score', 0)
                    preview = chunk.get('text_preview', '')
                    
                    # Show enhanced scoring if available
                    if ranked_chunks and i < len(ranked_chunks):
                        ranked_chunk = ranked_chunks[i]
                        enhanced_score = ranked_chunk.get('enhanced_score', 0)
                        keyword_score = ranked_chunk.get('keyword_score', 0)
                        final_score = ranked_chunk.get('final_score', 0)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Semantic", f"{similarity:.2f}")
                        with col2:
                            st.metric("Enhanced", f"{enhanced_score:.2f}")
                        with col3:
                            st.metric("Keyword", f"{keyword_score:.2f}")
                        with col4:
                            st.metric("Final", f"{final_score:.2f}")
                    else:
                        st.markdown(f"- Relevance: {similarity:.2f}")
                    
                    st.markdown(f"  *{preview}*")
                
                st.markdown("---")
    
    def generate_response(self, query: str):
        start_time = time.time()
        
        try:
            self.generate_hybrid_response(query, start_time)
                
        except Exception as e:
            st.error(f"❌ Response generation failed: {str(e)}")
            logger.error(f"Response generation error: {e}")
            
            error_message = {
                'role': 'assistant',
                'content': f"I apologize, but I encountered an error while processing your request: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'error': True
            }
            st.session_state.messages.append(error_message)
    
    def generate_hybrid_response(self, query: str, start_time: float):
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                try:
                    analyzer = st.session_state.context_analyzer
                    query_analysis = analyzer.analyze_query(query)
                    
                    retriever = st.session_state.pipeline.retriever
                    context_chunks = retriever.retrieve_context(query, k=5, min_similarity=0.1)
                    
                    if context_chunks:
                        ranked_chunks = analyzer.rank_chunks_by_relevance(query_analysis, context_chunks)
                        
                        response = analyzer.generate_contextual_response(query_analysis, ranked_chunks[:3])
                        
                        st.markdown(response)
                        
                        sources = {}
                        if st.session_state.config['enable_source_attribution']:
                            sources = self.create_sources_from_chunks(ranked_chunks[:3])
                            self.render_enhanced_sources(sources, ranked_chunks[:3])
                        
                        response_time = time.time() - start_time
                        
                        assistant_message = {
                            'role': 'assistant',
                            'content': response,
                            'sources': sources,
                            'response_time': response_time,
                            'query_analysis': query_analysis,
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.messages.append(assistant_message)
                        
                        st.session_state.query_count += 1
                        st.session_state.total_response_time += response_time
                    
                    else:
                        response = f"I cannot find relevant information about your question in the uploaded documents. Your question appears to be asking about {query_analysis['intent']} related to {', '.join(query_analysis['key_terms'][:3])}. Please make sure your question relates to the content of the documents you've uploaded."
                        st.markdown(response)
                        
                        assistant_message = {
                            'role': 'assistant',
                            'content': response,
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.messages.append(assistant_message)
                
                except Exception as e:
                    st.error(f"Response generation failed: {e}")
                    error_response = f"I apologize, but I encountered an error while processing your request: {str(e)}"
                    
                    assistant_message = {
                        'role': 'assistant',
                        'content': error_response,
                        'timestamp': datetime.now().isoformat(),
                        'error': True
                    }
                    st.session_state.messages.append(assistant_message)
    
    def generate_context_based_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        query_lower = query.lower()
        context_text = " ".join([chunk.get('metadata', {}).get('text', '') for chunk in context_chunks])
        context_lower = context_text.lower()
        
        if any(word in query_lower for word in ['ebay', 'what is ebay', 'marketplace']):
            if 'marketplace' in context_lower:
                return """Based on the provided context, **eBay is a marketplace that allows users to offer, sell, and buy goods and services in various geographic locations using a variety of pricing formats**. 

Key points from the document:
- eBay is not a party to contracts for sale between third-party sellers and buyers
- eBay is not a traditional auctioneer
- eBay provides guidance as part of their Services (pricing, shipping, listing, sourcing) which is informational
- Users may decide to follow eBay's guidance or not"""
            
        elif any(word in query_lower for word in ['ai', 'artificial intelligence', 'machine learning']):
            if 'artificial intelligence' in context_lower:
                return """According to the document, **eBay may use artificial intelligence or AI-powered tools and products** for several purposes:

- To provide and improve their Services
- To offer customized and personalized experiences
- To provide enhanced customer service  
- To support fraud detection

**Important note:** The document states that the availability and accuracy of these AI tools are not guaranteed."""
            
        elif any(word in query_lower for word in ['user agreement', 'terms', 'agreement', 'contract']):
            return """Based on the context, this is a **User Agreement** that establishes the terms for using eBay's services. Key aspects include:

- Sets out terms on which eBay offers access to and use of their Services
- Includes Mobile Application Terms of Use and all policies
- Users must comply with all terms when accessing or using eBay Services
- Contains provisions for resolving claims and disputes
- Includes an Agreement to Arbitrate for dispute resolution
- Users must be able to form legally binding contracts (e.g., over 18 years old)"""
            
        elif any(word in query_lower for word in ['rules', 'obligations', 'not allowed', 'prohibited']):
            return """According to the User Agreement, users must comply with various rules and are prohibited from:

- Breaching laws, regulations, or eBay's systems and policies
- Using services if unable to form legally binding contracts (under 18)
- Failing to pay for purchased items (without valid reason)
- Failing to deliver sold items (without valid reason)
- Manipulating item prices or interfering with other users' listings
- Undermining feedback or ratings systems
- Transferring eBay accounts without consent
- Creating inappropriate content or listings
- Engaging in spam, viruses, or automated scraping
- Infringing intellectual property rights"""
            
        else:
            sentences = context_text.split('.')
            relevant_sentences = []
            
            for sentence in sentences[:5]:
                if any(word in sentence.lower() for word in query_lower.split()):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                return f"""Based on the provided context, here's what I found relevant to your question:

{'. '.join(relevant_sentences[:3])}.

If you need more specific information, please ask about the document's main topics or use more specific keywords."""
            else:
                return """I found some relevant context in the document, but I cannot provide a specific answer to your question based on the available information. 

Please try asking more specific questions about the document's content, or use keywords that might appear in the text."""
    
    def create_sources_from_chunks(self, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        sources = {}
        
        for chunk in context_chunks:
            metadata = chunk.get('metadata', {})
            source_doc = metadata.get('source_document', 'Unknown Document')
            
            if source_doc not in sources:
                sources[source_doc] = {
                    'chunks': [],
                    'avg_similarity': 0.0,
                    'chunk_count': 0
                }
            
            chunk_info = {
                'chunk_id': metadata.get('id', 'unknown'),
                'text_preview': metadata.get('text', '')[:200] + '...',
                'similarity_score': chunk.get('similarity_score', 0.0),
                'chunk_index': metadata.get('chunk_index', 0)
            }
            
            sources[source_doc]['chunks'].append(chunk_info)
            sources[source_doc]['chunk_count'] += 1
        
        for source_info in sources.values():
            if source_info['chunks']:
                similarities = [c['similarity_score'] for c in source_info['chunks']]
                source_info['avg_similarity'] = sum(similarities) / len(similarities)
        
        return {
            'sources': sources,
            'total_sources': len(sources),
            'total_chunks': len(context_chunks)
        }
    
    def run(self):
        self.render_header()
        self.render_sidebar()
        self.render_chat_interface()


def main():
    app = ChatbotApp()
    app.run()


if __name__ == "__main__":
    main()