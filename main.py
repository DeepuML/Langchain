"""
main.py — RAG Chatbot with LangChain Agent and Multi-Tool Support

This script implements a Retrieval-Augmented Generation (RAG) chatbot that:
  1. Loads a text document and indexes it into a FAISS vector store.
  2. Uses a RetrievalQA chain to fetch relevant context for every user query.
  3. Passes the query and retrieved context to a LangChain agent equipped with
     five tools: Calculator, Summarizer, Wikipedia, Translator, and CodeExplainer.
  4. Maintains conversation history across turns via ConversationBufferMemory.

Configuration:
  Set EURI_API_KEY in a .env file (see .env.example). The data file path can
  be overridden with the DATA_FILE environment variable.

Usage:
  python main.py
"""

import os
import json

import numpy as np
import requests
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
from langchain.embeddings.base import Embeddings

import wikipedia

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

EURI_API_KEY = os.getenv("EURI_API_KEY", "")

# Path to the document used for the RAG knowledge base.
# Override with the DATA_FILE environment variable if the file lives elsewhere.
DATA_FILE = os.getenv(
    "DATA_FILE",
    os.path.join(os.path.dirname(__file__), "data", "google.txt"),
)

# ---------------------------------------------------------------------------
# EURI API helpers
# ---------------------------------------------------------------------------

def euri_embed(text: str) -> np.ndarray:
    """Generate a dense embedding vector for *text* using the EURI embeddings API.

    Args:
        text: The input string to embed.

    Returns:
        A NumPy array containing the embedding vector.

    Raises:
        ValueError: If the API returns a non-200 status code, the response
                    cannot be decoded as JSON, or the expected 'data' key is
                    absent from the response body.
    """
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}",
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small",
    }

    response = requests.post(url, headers=headers, json=payload)

    try:
        data = response.json()
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from the EURI embeddings API response.")

    if response.status_code != 200:
        raise ValueError(
            f"EURI embeddings API returned status {response.status_code}: {data}"
        )

    if "data" not in data:
        raise ValueError(
            f"Expected 'data' key not found in EURI embeddings response: {data}"
        )

    return np.array(data["data"][0]["embedding"])


def euri_chat(messages: list) -> str:
    """Send a chat completion request to the EURI API and return the reply text.

    Args:
        messages: A list of message dicts in OpenAI format, e.g.
                  [{"role": "user", "content": "Hello"}].

    Returns:
        The assistant's reply as a plain string.
    """
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}",
    }
    payload = {
        "model": "gpt-4.1-nano",
        "messages": messages,
        "max_tokens": 400,
        "temperature": 0.4,
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Custom LangChain LLM wrapper
# ---------------------------------------------------------------------------

class EuriLLM(LLM):
    """A LangChain LLM wrapper around the EURI chat completions API.

    This class lets EuriLLM be used as a drop-in replacement for any
    LangChain-compatible LLM inside chains, agents, and QA pipelines.
    """

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        """Invoke the EURI chat API with a single prompt string.

        Used by LLMChain and similar single-turn interfaces.

        Args:
            prompt: The user prompt to send to the model.
            stop:   Optional list of stop sequences (unused by EURI API).

        Returns:
            The model's response as a string.
        """
        return euri_chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ])

    def _generate(self, prompts: list, stop=None, **kwargs) -> LLMResult:
        """Invoke the EURI chat API for a batch of prompts.

        Used by agents and other batch-oriented interfaces.

        Args:
            prompts: A list of prompt strings.
            stop:    Optional list of stop sequences (unused by EURI API).

        Returns:
            An LLMResult containing one Generation per input prompt.
        """
        generations = []
        for prompt in prompts:
            output = self._call(prompt)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)

    @property
    def _identifying_params(self) -> dict:
        """Return model identification parameters (used for caching/logging)."""
        return {}

    @property
    def _llm_type(self) -> str:
        """Return a string identifier for this LLM type."""
        return "euri-llm"


# ---------------------------------------------------------------------------
# Custom LangChain Embeddings wrapper
# ---------------------------------------------------------------------------

class EuriEmbeddings(Embeddings):
    """A LangChain Embeddings wrapper around the EURI embeddings API.

    Enables EuriEmbeddings to be used with any LangChain vector store
    (e.g., FAISS) without additional configuration.
    """

    def embed_documents(self, texts: list) -> list:
        """Embed a list of documents.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        return [euri_embed(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list:
        """Embed a single query string.

        Args:
            text: The query string to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        return euri_embed(text).tolist()


# ---------------------------------------------------------------------------
# Load document and build FAISS index
# ---------------------------------------------------------------------------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Split the document into 500-character chunks to stay within embedding limits.
chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
documents = [Document(page_content=chunk) for chunk in chunks]

embedding_model = EuriEmbeddings()

# Build an in-memory FAISS index from the document chunks.
faiss_index = FAISS.from_texts(
    texts=[doc.page_content for doc in documents],
    embedding=embedding_model,
)

retriever = faiss_index.as_retriever()

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def calculator_tool(query: str) -> dict:
    """Evaluate an arithmetic expression and return the numeric result.

    Args:
        query: A string containing a valid Python arithmetic expression,
               e.g. "128 * 256" or "3.14 * (5 ** 2)".

    Returns:
        A dict with key "result" (the computed value as a string) on success,
        or key "error" (the exception) on failure.
    """
    try:
        result = str(eval(query))  # noqa: S307 — intentional for numeric expressions
        return {"result": result}
    except Exception as e:
        return {"error": e}


def summarizer_tool(text: str) -> str:
    """Summarize arbitrary text using the EURI chat API.

    Args:
        text: The text to summarize.

    Returns:
        A concise summary produced by the language model.
    """
    return euri_chat([
        {"role": "system", "content": "You summarize content concisely."},
        {"role": "user", "content": f"Summarize:\n{text}"},
    ])


def wikipedia_tool(query: str) -> str:
    """Fetch a short Wikipedia summary for *query*.

    Args:
        query: The search term to look up on Wikipedia.

    Returns:
        A 3-sentence Wikipedia summary, or an error message if the lookup fails.
    """
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Error: {e}"


def translate_tool(input_text: str) -> str:
    """Translate text into a target language using the EURI chat API.

    Expected input format: ``"<text> || <target language>"``.
    For example: ``"Hello world || French"``.

    Args:
        input_text: Combined text and target language separated by " || ".

    Returns:
        The translated text, or an error message if the input format is invalid.
    """
    if " || " not in input_text:
        return "Invalid input format. Use: Text || Language"

    text, target_language = input_text.split(" || ", maxsplit=1)
    return euri_chat([
        {"role": "system", "content": f"You translate content to {target_language}."},
        {"role": "user", "content": f"Translate:\n{text}"},
    ])


def explain_code_tool(code: str) -> str:
    """Explain what a code snippet does using the EURI chat API.

    Args:
        code: The source code snippet to explain.

    Returns:
        A plain-language explanation of the code.
    """
    return euri_chat([
        {"role": "system", "content": "You explain code clearly and concisely."},
        {"role": "user", "content": f"Explain this code:\n{code}"},
    ])


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description=(
            "MUST be used for ANY calculation request. "
            "Always use this tool to compute math expressions."
        ),
    ),
    Tool(
        name="Summarizer",
        func=summarizer_tool,
        description="Summarizes any text provided.",
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia_tool,
        description=(
            "Searches Wikipedia and returns a summary. "
            "Input should be the search term."
        ),
    ),
    Tool(
        name="Translator",
        func=translate_tool,
        description=(
            "Translates text into a target language. "
            "Input format: 'Text || Language'."
        ),
    ),
    Tool(
        name="CodeExplainer",
        func=explain_code_tool,
        description="Explains what a code snippet does.",
    ),
]

# Retain the full conversation history so the agent can refer back to earlier turns.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = EuriLLM()

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# Combine retrieval-based context with the agent's tool-augmented reasoning.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    # Step 1 — retrieve relevant document passages for the current query.
    retrieved_answer = qa_chain({"query": user_input})["result"]

    # Step 2 — let the agent decide whether to use tools, memory, or the
    #           retrieved context to formulate the final response.
    final_response = agent.invoke(
        f"{user_input}\nRetrieved Info: {retrieved_answer}"
    )

    # Debug output: show the full conversation history for inspection.
    print("\n[DEBUG] Memory so far:")
    for m in memory.chat_memory.messages:
        print(f"  {m.type.upper()}: {m.content}")

    print(f"\nBot: {final_response}\n")
