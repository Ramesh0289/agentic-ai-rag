# Agentic AI RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built for the Appening AI Engineer interview assignment.

The chatbot answers questions using the **Agentic AI eBook** as its knowledge source. It uses **Pinecone** for vector search, **LangGraph** to orchestrate retrieval and generation, **local sentence-transformer embeddings**, and **Ollama** for local LLM generation.

The project is designed to run without OpenAI API credits.

## 1. Project Overview

The system follows this flow:

```text
Agentic AI eBook PDF
        |
        v
PDF text extraction
        |
        v
Text chunking
        |
        v
Local embeddings
all-MiniLM-L6-v2
        |
        v
Pinecone vector database
        |
        v
LangGraph
  Retrieve -> Generate
        |
        v
Ollama / TinyLlama
        |
        v
FastAPI /chat
        |
        v
Answer + Retrieved Context + Confidence Score
```

The chatbot is instructed to answer only from the retrieved document context. If the document does not contain enough information, it should refuse rather than use unrelated outside knowledge.

## 2. Technologies

- Python
- LangGraph
- Pinecone
- FastAPI
- Ollama
- TinyLlama
- Sentence Transformers
- `all-MiniLM-L6-v2`
- PyPDF
- Pydantic Settings

## 3. Why Local Models?

The reference assignment suggests OpenAI embeddings and an OpenAI chat model. This implementation uses local alternatives so the project can be run without OpenAI API credits.

### Embeddings

```text
Model: all-MiniLM-L6-v2
Vector dimension: 384
```

### LLM

```text
Ollama
Model: tinyllama
```

TinyLlama was selected because the development machine has limited RAM. A larger local model such as `llama3.2:3b` was tested but could not reliably load with the available memory.

## 4. Project Structure

```text
agentic-ai-rag/
|
├── data/
│   └── Ebook-Agentic-AI.pdf
|
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── embeddings.py
│   ├── graph.py
│   ├── ingestion.py
│   └── vector_store.py
|
├── scripts/
│   ├── ingest.py
│   └── test_queries.py
|
├── tests/
│   └── test_api.py
|
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## 5. Prerequisites

- Python 3.10+
- Pinecone account and API key
- Ollama installed
- Internet connection for installing Python packages and downloading the embedding model

The project was tested with Python 3.13.

Install Ollama from the official Ollama website:

https://ollama.com/

Then download the local model:

```powershell
ollama pull tinyllama
```

Verify it:

```powershell
ollama list
```

## 6. Setup

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## 7. Environment Variables

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Open `.env` and add your Pinecone API key:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag

LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=local

OLLAMA_MODEL=tinyllama
EMBEDDING_MODEL=all-MiniLM-L6-v2

TOP_K=3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RELEVANCE_THRESHOLD=0.35

PDF_PATH=data/Ebook-Agentic-AI.pdf
```

Never commit `.env` or any real API key to GitHub.

## 8. Add the Knowledge Source

Place the Agentic AI eBook at:

```text
data/Ebook-Agentic-AI.pdf
```

The project uses this local file during ingestion.

## 9. Document Ingestion

Run ingestion from the project root:

```powershell
python -m scripts.ingest
```

The ingestion process:

1. Reads the PDF.
2. Extracts text.
3. Splits the text into chunks.
4. Generates local 384-dimensional embeddings.
5. Creates/uses the Pinecone index.
6. Stores the vectors and metadata in Pinecone.

The completed ingestion for this project produced:

```text
119 chunks
```

## 10. LangGraph Workflow

The LangGraph workflow is intentionally simple:

```text
START
  |
  v
retrieve
  |
  v
generate
  |
  v
END
```

### Retrieve node

The retrieve node searches Pinecone for the top relevant chunks for the user's question.

### Generate node

The generate node sends the retrieved context and question to Ollama.

The prompt instructs the model to:

- use only the supplied context
- avoid unsupported claims
- state that it cannot answer when the document does not contain enough information

## 11. Run the Test Queries

Run:

```powershell
python -m scripts.test_queries
```

The project includes questions covering:

1. What is Agentic AI according to the eBook?
2. How do AI agents differ from traditional automation systems?
3. What are the core components of an Agentic Architecture?
4. What role does memory play in Agentic AI workflows?
5. What real-world industry use cases are discussed in the eBook?
6. Who won the 2022 FIFA World Cup?

The final question is an out-of-scope test. The chatbot should not answer it from outside knowledge.

### Observed test behavior

The completed test run produced answers with retrieval scores such as:

```text
Test 1: 0.82
Test 2: 0.68
Test 3: 0.65
Test 4: 0.66
Test 5: 0.43
Test 6: 0.00
```

For the FIFA question, the chatbot returned:

```text
I cannot answer this question based on the provided document.
```

The scores are used as a simple retrieval/relevance indicator. They should not be interpreted as calibrated probabilities.

## 12. Run the FastAPI Application

Start the API:

```powershell
python -m uvicorn src.app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## 13. Chat API

Endpoint:

```text
POST /chat
```

Example request:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

The response contains:

```json
{
  "query": "What is Agentic AI according to the eBook?",
  "final_answer": "...",
  "retrieved_context_chunks": [
    "...",
    "...",
    "..."
  ],
  "confidence_score": 0.82
}
```

The exact answer and score depend on the retrieved chunks and local model output.

## 14. Grounding Behavior

The main goal is to keep answers grounded in the Agentic AI eBook.

For example:

```text
Question:
Who won the 2022 FIFA World Cup?
```

Expected behavior:

```text
I cannot answer this question based on the provided document.
```

This prevents the chatbot from answering using unrelated general knowledge.

## 15. Testing the API

After starting FastAPI, open:

```text
http://127.0.0.1:8000/docs
```

Use `POST /chat` and test:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

Then test an out-of-scope question:

```json
{
  "query": "Who won the 2022 FIFA World Cup?"
}
```

## 16. Troubleshooting

### Ollama model does not load

Check installed models:

```powershell
ollama list
```

Make sure TinyLlama is installed:

```powershell
ollama pull tinyllama
```

Test it directly:

```powershell
ollama run tinyllama
```

### `ModuleNotFoundError: No module named 'src'`

Run project scripts as modules from the project root:

```powershell
python -m scripts.ingest
```

instead of:

```powershell
python scripts\ingest.py
```

### Windows memory error

Local embedding models and LLMs require RAM. On machines with limited memory, close unnecessary applications before running the chatbot.

### Pinecone errors

Check:

```env
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=agentic-ai-rag
```

Make sure the API key is valid and the machine has internet access.

## 17. Security

- `.env` must not be committed.
- API keys must not be placed in source code.
- `.env.example` contains placeholders only.
- The PDF is treated as the knowledge source for the chatbot.
- The LLM is instructed not to answer beyond retrieved context.

## 18. Assignment Requirements Covered

| Requirement | Implementation |
|---|---|
| Python | Python project |
| PDF ingestion | PyPDF |
| Chunking | 1000 characters / 200 overlap |
| Embeddings | all-MiniLM-L6-v2 |
| Vector database | Pinecone |
| LangGraph | Retrieve → Generate |
| Grounded generation | Context-only prompt |
| Confidence/relevance score | Returned in API response |
| API | FastAPI |
| Testing | 6 sample queries |
| Out-of-scope handling | Refusal when context is insufficient |
| Local LLM | Ollama / TinyLlama |

## 19. Notes

This is a focused interview-assignment implementation rather than a production-scale RAG platform. The design intentionally keeps the architecture small and easy to understand:

```text
Ingestion
    |
Pinecone
    |
LangGraph
    |
Ollama
    |
FastAPI
```

The local-model setup avoids OpenAI API usage and therefore does not require OpenAI credits.
