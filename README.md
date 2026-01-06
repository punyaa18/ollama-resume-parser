# AI Resume Parser & Candidate Evaluator with Advanced RAG

A web-based application that uses AI (Ollama + Llama3) with advanced Retrieval-Augmented Generation (RAG) to parse resumes and evaluate candidates against job descriptions.

## Features

- 📤 Upload PDF/TXT resumes with automatic parsing
- 🤖 **Advanced RAG System** with multiple retrieval strategies (Similarity, MMR, Hybrid)
- 📊 Skill matching and rating (0-10) for each resume
- 🎯 **RAG Strategies**: Similarity, MMR, Hybrid with Query Expansion
- 🔍 **Query Expansion** for better retrieval
- 🎨 **Configurable K** (documents to retrieve)
- 🌐 Web-based interface with modern UI
- 📋 Resume management with individual evaluation options

## Prerequisites

- Python 3.8+
- Ollama installed with:
  - `llama3` model
  - `nomic-embed-text` embeddings

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

4. Pull required models:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

## Usage

1. **Start the web application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to `http://localhost:5000`

3. **Upload resumes** using the file upload section - they'll be automatically parsed and stored

4. **View uploaded resumes** in the list with individual "Evaluate" buttons

5. **Evaluate resumes** in multiple ways:
   - **Automatic**: Upload triggers immediate evaluation
   - **Individual**: Click "Evaluate" button next to any resume
   - **Bulk**: Click "Evaluate All Resumes" for batch processing

6. **View results** in the evaluation section with detailed AI analysis

## Advanced RAG System

This application implements state-of-the-art Retrieval-Augmented Generation (RAG) techniques:

### Retrieval Strategies

- **Similarity Search**: Standard semantic similarity matching
- **MMR (Maximal Marginal Relevance)**: Balances relevance and diversity to avoid redundancy
- **Hybrid**: Combines MMR with query expansion for optimal results

### Features

- **Query Expansion**: Automatically generates additional search queries based on job requirements
- **Configurable K**: Adjust number of documents retrieved (2-10)
- **Real-time Configuration**: Change RAG settings without restarting
- **Evaluation Metrics**: Shows which strategy and parameters were used

### How RAG Works

1. **Ingestion**: Resumes are chunked and embedded using Ollama's nomic-embed-text
2. **Retrieval**: Multiple strategies find most relevant resume sections
3. **Augmentation**: Retrieved content is combined with job description
4. **Generation**: Llama3 provides comprehensive candidate evaluation

## File Structure

```
├── app.py                 # Flask web application
├── evaluator.py           # AI evaluation logic
├── ingest.py              # Resume ingestion script
├── job_description.txt    # Job requirements
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Web interface
├── resumes/               # Uploaded resume files
└── vector_store/          # FAISS vector database
```

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload resume file (automatically parsed and stored)
- `GET /resumes` - List uploaded resumes
- `GET /rag-config` - Get RAG configuration options
- `POST /evaluate` - Run AI evaluation on all resumes with RAG parameters
- `POST /evaluate/<filename>` - Run AI evaluation on specific resume with RAG parameters

### RAG Parameters

All evaluation endpoints support these query parameters:

- `strategy`: Retrieval strategy (`similarity`, `mmr`, `compression`, `hybrid`)
- `k`: Number of documents to retrieve (2-10)
- `expansion`: Enable query expansion (`true`/`false`)

## Technologies Used

- **Backend**: Python, Flask
- **AI**: Ollama (Llama3, Nomic Embeddings)
- **Vector DB**: FAISS
- **Frontend**: HTML, CSS, JavaScript
- **Document Processing**: LangChain, PyPDF