# 🦜🔗 LangChain RAG Agent

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-orange)](https://faiss.ai/)

A production-ready **Retrieval-Augmented Generation (RAG) chatbot** built with LangChain, integrating a FAISS vector store, a custom EURI LLM, and a multi-tool conversational agent. Includes a collection of Jupyter notebooks covering LangChain fundamentals — from LLMs and chat models to chains, embeddings, and structured output.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Module Descriptions](#-module-descriptions)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This repository serves two purposes:

1. **RAG Chatbot (`main.py`)** — A fully functional conversational AI system that:
   - Ingests a text document into a **FAISS** vector store
   - Uses **RetrievalQA** to fetch relevant context for every user query
   - Passes the query and retrieved context to a **LangChain agent** equipped with five tools
   - Maintains conversation history via **ConversationBufferMemory**

2. **Learning Notebooks (`Models/`)** — A structured set of Jupyter notebooks demonstrating core LangChain concepts step by step, from basic LLM calls to advanced parallel chains and structured output extraction.

---

## ✨ Key Features

- **RAG Pipeline** — Document chunking → FAISS indexing → similarity retrieval → LLM response
- **Custom EURI LLM** — Integrates the [Euron.one](https://euron.one) API as a drop-in LangChain LLM
- **Custom Embeddings** — `EuriEmbeddings` wraps the EURI embeddings endpoint for use with LangChain vector stores
- **Multi-Tool Agent** — Five built-in tools: Calculator, Summarizer, Wikipedia search, Translator, and Code Explainer
- **Persistent Memory** — `ConversationBufferMemory` retains the full conversation history across turns
- **Jupyter Notebooks** — Hands-on examples for LLMs, chat models, embeddings, LCEL chains, Pydantic, and structured output

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| LLM Orchestration | [LangChain](https://python.langchain.com/) |
| LLM Provider | [Euron.one EURI API](https://euron.one) (custom), OpenAI, Gemini, Anthropic, HuggingFace |
| Vector Store | [FAISS](https://faiss.ai/) via `langchain-community` |
| Embeddings | EURI Embeddings API, HuggingFace `sentence-transformers` |
| HTTP Client | [Requests](https://requests.readthedocs.io/) |
| Numerical Computing | [NumPy](https://numpy.org/) |
| Environment Config | [python-dotenv](https://pypi.org/project/python-dotenv/) |
| Knowledge Tool | [Wikipedia](https://pypi.org/project/wikipedia/) |
| Notebooks | [Jupyter](https://jupyter.org/) |

---

## 📁 Project Structure

```
Langchain/
├── main.py                  # RAG chatbot entry point (agent + memory + tools)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── data/
│   └── google.txt           # Sample document for the RAG pipeline
├── Models/                  # Learning notebooks and shared assets
│   ├── LLMS/
│   │   └── code.ipynb       # Basic LLM invocation (OpenAI)
│   ├── chatmodels/
│   │   └── code.ipynb       # ChatOpenAI, Anthropic, Gemini, HuggingFace
│   ├── Embeddingmodels/
│   │   └── code.ipynb       # HuggingFace embeddings + cosine similarity
│   ├── Chains/
│   │   ├── code.ipynb       # Simple LLM chains (LCEL)
│   │   ├── code3.ipynb      # Sequential and parallel chains
│   │   └── Project/         # End-to-end chain project with CSV data
│   ├── Pydantic/
│   │   └── code.ipynb       # Pydantic integration with LangChain
│   ├── structured_outptu/
│   │   └── code.ipynb       # Structured output via TypedDict schemas
│   ├── data/
│   │   └── google.txt       # Sample data shared with notebooks
│   └── main.py              # Standalone RAG agent (mirrors root main.py)
├── LICENSE
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9 or higher
- A valid [Euron.one EURI API key](https://euron.one) *(required for the RAG chatbot)*
- *(Optional)* API keys for OpenAI, Google Gemini, Anthropic, or HuggingFace *(required for specific notebooks)*

### 1. Clone the repository

```bash
git clone https://github.com/DeepuML/Langchain.git
cd Langchain
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Notebook-only dependencies** (install as needed):
> ```bash
> pip install langchain-openai langchain-google-genai langchain-anthropic
> pip install langchain-huggingface sentence-transformers
> pip install wikipedia
> ```

---

## ⚙️ Configuration

Copy the environment variable template and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required – EURI API key for the RAG chatbot (main.py)
EURI_API_KEY=your_euri_api_key_here

# Optional – required only for the corresponding notebooks
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.

The `main.py` script reads the API key via:

```python
import os
from dotenv import load_dotenv
load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")
```

---

## 💬 Usage

### Running the RAG Chatbot

```bash
python main.py
```

The chatbot will:
1. Load and chunk `data/google.txt` into a FAISS index
2. Start an interactive loop — type your question and press **Enter**
3. Retrieve relevant document passages, then let the agent answer using its tools
4. Type `exit` or `quit` to stop

**Example session:**

```
You: What does Google do?
[Agent reasoning...]
Bot: Google is a multinational technology company...

You: Calculate 128 * 256
[Agent uses Calculator tool]
Bot: 32768

You: exit
```

### Running the Jupyter Notebooks

```bash
jupyter notebook Models/
```

Open any notebook inside the `Models/` sub-directories to follow the guided examples.

---

## 📦 Module Descriptions

### `main.py` — RAG Chatbot

| Component | Description |
|---|---|
| `euri_embed(text)` | Calls the EURI embeddings endpoint and returns a NumPy vector |
| `euri_chat(messages)` | Calls the EURI chat completions endpoint and returns the response string |
| `EuriLLM` | LangChain `LLM` subclass wrapping `euri_chat` for use in chains and agents |
| `EuriEmbeddings` | LangChain `Embeddings` subclass wrapping `euri_embed` for use with FAISS |
| `calculator_tool` | Safely evaluates arithmetic expressions |
| `summarizer_tool` | Summarizes arbitrary text via the EURI chat API |
| `wikipedia_tool` | Fetches a Wikipedia summary for a search term |
| `translate_tool` | Translates text into a target language via the EURI chat API |
| `explain_code_tool` | Explains a code snippet via the EURI chat API |

### `Models/` — Learning Notebooks

| Notebook | Topics Covered |
|---|---|
| `LLMS/code.ipynb` | Basic LLM invocation with `OpenAI` |
| `chatmodels/code.ipynb` | `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`, `ChatHuggingFace` |
| `Embeddingmodels/code.ipynb` | HuggingFace embeddings, cosine similarity search |
| `Chains/code.ipynb` | Simple LCEL chains: `prompt \| model \| parser` |
| `Chains/code3.ipynb` | Sequential chains with `RunnableLambda`, parallel chains with `RunnableParallel` |
| `Pydantic/code.ipynb` | Pydantic model integration with LangChain |
| `structured_outptu/code.ipynb` | Structured output using `TypedDict` and `with_structured_output` |

---

## 🔌 API Reference

### EURI Embeddings Endpoint

| Field | Value |
|---|---|
| URL | `https://api.euron.one/api/v1/euri/embeddings` |
| Method | `POST` |
| Model | `text-embedding-3-small` |
| Auth | `Bearer <EURI_API_KEY>` |

**Request body:**
```json
{ "input": "<text>", "model": "text-embedding-3-small" }
```

**Response shape:**
```json
{ "data": [{ "embedding": [0.01, -0.02, ...] }] }
```

### EURI Chat Completions Endpoint

| Field | Value |
|---|---|
| URL | `https://api.euron.one/api/v1/euri/chat/completions` |
| Method | `POST` |
| Model | `gpt-4.1-nano` |
| Auth | `Bearer <EURI_API_KEY>` |

**Request body:**
```json
{
  "model": "gpt-4.1-nano",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 400,
  "temperature": 0.4
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feat/your-feature`
3. **Commit** your changes using [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add streaming support to EuriLLM
   fix: resolve hardcoded data path in main.py
   docs: update API reference in README
   refactor: extract tool definitions into tools.py
   ```
4. **Push** your branch and open a **Pull Request**

Please ensure:
- New code follows the existing style
- The chatbot runs without errors (`python main.py`)
- Notebooks execute cleanly from top to bottom

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

*Built with ❤️ by [Deependra Gangwar](https://github.com/DeepuML)*
