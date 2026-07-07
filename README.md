# LaatGPT

A RAG (Retrieval-Augmented Generation) chatbot that lets you upload a PDF and ask questions about its contents. Built with Streamlit, LangChain, FAISS, and OpenRouter.

## How it works

1. Upload a PDF from the sidebar.
2. The document is parsed with `pdfplumber` and split into overlapping chunks.
3. Chunks are embedded via an OpenRouter embedding model and stored in a FAISS vector store.
4. On each question, the most relevant chunks are retrieved (MMR search) and passed as context to an OpenRouter LLM, which streams its answer back in the chat UI.

## Project structure

| File | Purpose |
|---|---|
| [app.py](app.py) | Streamlit entry point / UI |
| [document_processor.py](document_processor.py) | PDF text extraction and chunking |
| [embeddings.py](embeddings.py) | OpenRouter embeddings wrapper (LangChain `Embeddings` interface) |
| [vector_store.py](vector_store.py) | FAISS vector store build/retriever helper |
| [rag_chain.py](rag_chain.py) | Prompt template + retrieval chain (LangChain LCEL) |
| [chat_ui.py](chat_ui.py) | Chat message rendering/styling helpers |
| [config.py](config.py) | Environment-based settings |

## Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key

## Setup

1. Clone the repository and enter the project directory.

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following variables:

   ```bash
   OPENROUTER_API_KEY=your_openrouter_api_key
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   OPENROUTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
   OPENROUTER_LLM_MODEL=nvidia/nemotron-nano-9b-v2:free
   ```

   Only `OPENROUTER_API_KEY` is required; the others fall back to the defaults shown above. When deploying to Streamlit Community Cloud, set these in `st.secrets` instead.

## Running the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`), upload a PDF, and start chatting.

## Tech stack

- [Streamlit](https://streamlit.io/) — UI
- [LangChain](https://www.langchain.com/) — orchestration (text splitting, prompt templates, LCEL chains)
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF text extraction
- [OpenRouter](https://openrouter.ai/) — embeddings and LLM inference
