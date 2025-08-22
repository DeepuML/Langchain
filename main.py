import os
import numpy as np
import json
import requests

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQA

EURI_API_KEY = "euri-02bada99e8bed5fd1250d365d93c30abb89c621ce3d1a3a8e7a351f70c6aefe0"

def euri_embed(text):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    
    # Try parsing the JSON
    try:
        data = response.json()
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from EURI API response.")

    # Check for error in the API response
    if response.status_code != 200:
        raise ValueError(f"EURI API returned status {response.status_code}: {data}")

    if 'data' not in data:
        raise ValueError(f"Expected 'data' key not found in response: {data}")

    embedding = np.array(data['data'][0]['embedding'])
    return embedding


def euri_chat(message):
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Write a poem about artificial intelligence"
            }
        ],
        "model": "gpt-4.1-nano",
        "messages":message,
        "max_tokens": 400,
        "temperature": 0.4
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']


from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation

class EuriLLM(LLM):
    def _call(self, prompt, stop=None, **kwargs) -> str:
        """Single prompt usage (e.g., LLMChain)"""
        return euri_chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])

    def _generate(self, prompts, stop=None, **kwargs) -> LLMResult:
        """Batch prompt usage (e.g., Agents)"""
        generations = []
        for prompt in prompts:
            output = self._call(prompt)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "euri-llm"

with open("D:\Project\Genai\Langchain\data\google.txt", "r", encoding="utf-8") as f:
    text = f.read()
    # print(text)



chunks = [text[i:i+500] for i in range(0, len(text), 500)]
documents = [Document(page_content=chunk) for chunk in chunks]

# print("Printing the chunks: ",chunks)

# print("Printing the documents: ",documents)
    
from langchain.embeddings.base import Embeddings

class EuriEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [euri_embed(t).tolist() for t in texts]

    def embed_query(self, text):
        return euri_embed(text).tolist()

embedding_model = EuriEmbeddings()

faiss_index = FAISS.from_texts(
    texts=[doc.page_content for doc in documents],
    embedding=embedding_model
)

retriever = faiss_index.as_retriever()

def calculator_tool(query):
    """evaluate the arithmetic expression"""
    try:
        result = str(eval(query))
        return {"result": result}
    except Exception as e:
        return {"error": e}      

def summarizer_tool(text):
    """Summarize any text"""
    return euri_chat([
        {"role": "system", "content": "You summarize content."},
        {"role": "user", "content": f"Summarize:\n{text}"}
    ])

import wikipedia

def wikipedia_tool(query):
    """Fetch a summary from Wikipedia."""
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return f"Error: {e}"
    
def translate_tool(text, target_language):
    """Translate text to the target language."""
    return euri_chat([
        {"role": "system", "content": f"You translate content to {target_language}."},
        {"role": "user", "content": f"Translate:\n{text}"}
    ]) 

    return "Invalid input format. Use: Text || Language"

def explain_code_tool(code):
    """Explain a piece of code."""
    return euri_chat([  
        {"role": "system", "content": "You explain code."},
        {"role": "user", "content": f"Explain this code:\n{code}"}
    ])
    prompt = [
        {"role": "system", "content": f"You translate text to {target_language}."},
        {"role": "user", "content": f"Translate this into {target_language}:\n{text}"}
    ]
    return euri_chat(prompt)



tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="MUST be used for ANY calculation request. Always use this tool to compute math expressions."

    ),
    Tool(
        name="Summarizer",
        func=summarizer_tool,
        description="Summarizes any text provided."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia_tool,
        description="Searches Wikipedia and returns a summary. Input should be the search term."
    ),
    Tool(
        name="Translator",
        func=translate_tool,
        description="Translates text into a target language. Input format: 'Text || Language'."
    ),
    Tool(
        name="CodeExplainer",
        func=explain_code_tool,
        description="Explains what a code snippet does."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = EuriLLM()

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break

    # Use RetrievalQA first
    retrieved_answer = qa_chain({"query": user_input})["result"]

    # Let the agent decide whether to use tools/memory
    final_response = agent.invoke(f"{user_input}\nRetrieved Info: {retrieved_answer}")

    print("\n[DEBUG] Memory so far:")
    for m in memory.chat_memory.messages:
        print(f"{m.type.upper()}: {m.content}")
    
    print(f"\nBot: {final_response}\n")
