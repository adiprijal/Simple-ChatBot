# Simple-ChatBot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from a PDF using Google Gemini, LangChain, and Chroma vector search.

## Overview

This project loads a PDF, splits it into semantic chunks, embeds those chunks with Google embedding models, stores them in Chroma, and answers user questions using a Gemini chat model with retrieval context.

## Key Features

- PDF-based question answering from local documents.
- Embedding model fallback logic for higher reliability.
- Retrieval-augmented responses using top-k relevant chunks.
- Simple interactive terminal chat loop.
- Clean startup validation for missing API keys and dependencies.

## Tech Stack

- Python 3.10+
- LangChain + LangChain Community
- Google Generative AI (Gemini models)
- Chroma vector store
- PyPDFLoader for document ingestion

## How It Works

1. Load `sample.pdf` from the project root.
2. Split PDF pages into chunks (`chunk_size=800`, `chunk_overlap=80`).
3. Generate embeddings (tries multiple candidate models).
4. Build an in-memory Chroma vector store from chunks.
5. For each user query:
   - Retrieve top 3 relevant chunks.
   - Send context + question to `gemini-2.5-flash`.
   - Return the final answer.

## Project Structure

```text
Simple-ChatBot/
|-- app.py
|-- requirements.txt
|-- sample.pdf
|-- .env
`-- README.md
```

## Quick Start

### 1) Clone the repository

```bash
git clone https://github.com/adiprijal/Simple-ChatBot.git
cd Simple-ChatBot
```

### 2) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 5) Add your PDF

Place your target file as `sample.pdf` in the root directory (or update the path in `app.py`).

### 6) Run the chatbot

```bash
python app.py
```

## Example Usage

```text
--- Chatbot Ready ---

Ask about the PDF (or 'exit'): What is the main topic?
Answer: <model response>

Ask about the PDF (or 'exit'): exit
```

## Configuration Notes

- Embedding candidates are currently:
  - `models/gemini-embedding-001`
  - `models/gemini-embedding-2-preview`
- If one embedding model fails, the app automatically tries the next.
- Retrieval uses `k=3` by default.
- Chat model is set to `gemini-2.5-flash` with `temperature=0.7`.

## Common Issues

### `GOOGLE_API_KEY not found`

Make sure `.env` exists in the root and contains a valid key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### `Run 'pip install langchain-classic'`

Install missing dependency:

```bash
pip install langchain-classic
```

### `File 'sample.pdf' not found`

Add `sample.pdf` to the root directory or change the path in `app.py`.

### Embedding initialization or quota errors

- Verify your key in Google AI Studio.
- Check API quota and billing limits.
- Retry later if service is temporarily unavailable.

## Suggested Next Improvements

- Add CLI arguments for custom PDF path and retrieval settings.
- Add persistent Chroma storage directory.
- Add source citation output per answer.
- Add unit tests for chunking and retrieval configuration.

## Author

[Adip Rijal](https://github.com/adiprijal)
