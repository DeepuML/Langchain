# 📚 Models — LangChain Learning Notebooks

This directory contains a structured collection of Jupyter notebooks that explore core LangChain concepts step by step. Each sub-directory focuses on a specific topic, progressing from basic LLM invocation to advanced chains and structured output.

---

## 📁 Directory Layout

```
Models/
├── LLMS/
│   └── code.ipynb           # Basic LLM invocation (OpenAI)
├── chatmodels/
│   └── code.ipynb           # Chat models: OpenAI, Anthropic, Gemini, HuggingFace
├── Embeddingmodels/
│   └── code.ipynb           # Sentence embeddings and cosine similarity
├── Chains/
│   ├── code.ipynb           # Simple LCEL chain (prompt | model | parser)
│   ├── code3.ipynb          # Sequential and parallel chains
│   └── Project/             # End-to-end chain project with real CSV data
├── Pydantic/
│   └── code.ipynb           # Pydantic model integration with LangChain
├── structured_outptu/
│   └── code.ipynb           # Structured output via TypedDict schemas
├── data/
│   └── google.txt           # Sample document used in RAG examples
└── main.py                  # Standalone RAG agent (mirrors root main.py)
```

---

## 📓 Notebook Summaries

### `LLMS/code.ipynb` — Basic LLM Usage
Demonstrates how to invoke a legacy OpenAI LLM directly using LangChain's `OpenAI` wrapper.

**Key concepts:** `OpenAI`, `llm.invoke`, environment variables via `dotenv`

---

### `chatmodels/code.ipynb` — Chat Model Integrations
Shows how to use four different chat model providers through a unified LangChain interface.

**Providers covered:**
| Provider | Class | Model |
|---|---|---|
| OpenAI | `ChatOpenAI` | `gpt-4o` |
| Anthropic | `ChatAnthropic` | `claude-2` |
| Google | `ChatGoogleGenerativeAI` | `gemini-1.5-flash` |
| HuggingFace | `ChatHuggingFace` + `HuggingFaceEndpoint` | `TinyLlama-1.1B-Chat` |

**Key concepts:** `chat.invoke`, `result.content`, model-specific parameters

---

### `Embeddingmodels/code.ipynb` — Embeddings and Similarity Search
Generates sentence embeddings with HuggingFace and computes cosine similarity to find the most relevant document for a query.

**Key concepts:** `HuggingFaceEmbeddings`, `embed_documents`, `embed_query`, `cosine_similarity`, `sentence-transformers/all-MiniLM-L6-v2`

---

### `Chains/code.ipynb` — Simple LangChain Expression Language (LCEL) Chains
Builds a basic chain using the pipe operator: `prompt | model | parser`.

**Key concepts:** `PromptTemplate`, `StrOutputParser`, LCEL pipe syntax, `chain.invoke`, `chain.get_graph().print_ascii()`

---

### `Chains/code3.ipynb` — Sequential and Parallel Chains
Demonstrates more advanced chain compositions:
- **Sequential chain** — output of one LLM feeds into the next prompt via `RunnableLambda`
- **Parallel chain** — two LLM branches run concurrently via `RunnableParallel`, then merge

**Key concepts:** `RunnableLambda`, `RunnableParallel`, multi-step reasoning

---

### `Chains/Project/` — End-to-End Chain Project
An applied project that uses a real e-commerce delivery dataset (`synthetic_ecommerce_delivery_data_large.csv`) with LangChain chains.

**Key concepts:** Data-driven prompting, CSV analysis, chained LLM reasoning

---

### `Pydantic/code.ipynb` — Pydantic with LangChain
Explores how Pydantic models integrate with LangChain for type-safe input/output handling.

**Key concepts:** Pydantic `BaseModel`, field validation, LangChain compatibility

---

### `structured_outptu/code.ipynb` — Structured Output Extraction
Uses `with_structured_output` to extract typed, schema-validated data from free-form text (e.g., product reviews).

**Key concepts:** `TypedDict`, `Annotated`, `Literal`, `Optional`, `with_structured_output`

**Example schema:**
```python
class Review(TypedDict):
    key_themes: Annotated[list[str], "Key themes from the review"]
    summary:    Annotated[str, "Brief summary"]
    sentiment:  Annotated[Literal["pos", "neg"], "Sentiment"]
    pros:       Annotated[Optional[list[str]], "Pros listed"]
    cons:       Annotated[Optional[list[str]], "Cons listed"]
    name:       Annotated[Optional[str], "Reviewer name"]
```

---

## ⚙️ Setup

Most notebooks require at least one API key. Copy the root `.env.example` to `.env` and populate the relevant keys:

```env
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...
EURI_API_KEY=...
```

Install notebook-specific dependencies as prompted at the top of each notebook, or run:

```bash
pip install langchain-openai langchain-google-genai langchain-anthropic
pip install langchain-huggingface sentence-transformers wikipedia
```

---

## 📄 License

MIT — see the root [LICENSE](../LICENSE) file.
