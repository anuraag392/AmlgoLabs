# RAG Chatbot - Amlgo Labs
Project Link and source code : https://github.com/anuraag392/AmlgoLabs

A Fine-Tuned RAG (Retrieval-Augmented Generation) Chatbot with Streaming Responses built for Amlgo Labs.

## Features

- 🤖 **Advanced RAG Pipeline**: Complete retrieval-augmented generation system
- 📚 **Document Processing**: Intelligent text cleaning, chunking, and validation
- 🔍 **Semantic Search**: FAISS-powered vector similarity search with reranking
- 💬 **Streaming Responses**: Real-time token-by-token response generation
- 🎯 **Source Attribution**: Automatic citation of relevant document sources
- 🎨 **Interactive UI**: Modern Streamlit-based chat interface
- ⚡ **Performance Optimized**: Efficient embedding generation and caching
- 🛡️ **Error Recovery**: Robust error handling with fallback mechanisms

## Architecture

The system consists of several key components:

1. **Document Processing Pipeline**
   - `DocumentProcessor`: Text cleaning and validation
   - `TextChunker`: Intelligent document segmentation
   - `EmbeddingGenerator`: Semantic embedding creation

2. **Retrieval System**
   - `VectorStore`: FAISS-based vector database
   - `SemanticRetriever`: Context retrieval with ranking

3. **Generation System**
   - `LLMManager`: Language model integration
   - `PromptTemplate`: Dynamic prompt engineering
   - `RAGPipeline`: End-to-end orchestration

4. **User Interface**
   - `ResponseStreamer`: Real-time response delivery
   - `ChatbotApp`: Streamlit web application

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run src/chatbot_app.py
```

## Usage

1. **Initialize System**: Click "🚀 Initialize System" in the sidebar
2. **Upload Documents**: Use the file uploader to add your knowledge base
3. **Process Documents**: Click "📥 Process Documents" to build the vector index
4. **Start Chatting**: Ask questions about your uploaded documents

## Configuration

The system supports various configuration options:

- **Model Settings**: Temperature, max tokens, context chunks
- **Retrieval Settings**: Similarity threshold, reranking options
- **UI Settings**: Streaming mode, source citations

## Testing

Run the test suite:
```bash
pytest src/test_*.py -v
```

## Performance

The system is optimized for:
- Fast document processing and indexing
- Efficient similarity search with FAISS
- Real-time streaming responses
- Memory-efficient embedding generation

## Requirements

- Python 3.8+
- 8GB+ RAM recommended
- GPU support optional (CUDA-compatible)

## License

Built for Amlgo Labs - All rights reserved.
