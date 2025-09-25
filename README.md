# RAG over PDFs (OpenAI + Streamlit)

A minimal RAG app to index PDFs into ChromaDB and answer questions using OpenAI.

## Features

- Upload single or multiple PDFs and index them into a persistent Chroma store
- OpenAI embeddings: `text-embedding-3-large`
- Chat completion via `ChatOpenAI` (default `gpt-4o`)
- Simple Streamlit UI

## Setup

1. Create a virtual environment and activate it (Windows PowerShell):
   ```ps1
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide your OpenAI API key either via environment or `.env` file in project root:
   - Environment (PowerShell):
     ```ps1
     $Env:OPENAI_API_KEY="sk-..."
     ```
   - Or create `.env` in the project root with:
     ```
     OPENAI_API_KEY=sk-...
     ```

## Run

From the project root:

```bash
streamlit run app/app.py
```

- Upload PDFs in the sidebar; click "Add uploaded to index" to embed and store.
- Ask a question in the main panel to retrieve context and get an answer.

## Data locations

- Uploaded files: `data/uploads/`
- Vector store (Chroma persistent client): `data/vector_store/`

## Notes

- To change models, edit them in the sidebar (embedding and chat models).
- Ensure you have sufficient OpenAI quota to avoid 429 errors.
