import os
import json
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


st.set_page_config(
    page_title="Indecimal AI Assistant",
    page_icon="🏗️",
    layout="centered"
)

def load_documents(docs_folder="docs"):
    documents = []
    for filepath in Path(docs_folder).glob("*.md"):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append({"filename": filepath.name, "content": content})
    return documents

def chunk_document(doc, chunk_size=100, overlap=20):
    chunks = []
    words = doc["content"].split()
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "filename": doc["filename"],
            "chunk_index": len(chunks),
            "text": chunk_text
        })
        start = end - overlap
    return chunks


@st.cache_resource
def build_rag_index():
    documents = load_documents()
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return embedder, index, all_chunks


def retrieve_chunks(query, embedder, index, all_chunks, top_k=3):
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [{
        "rank": i + 1,
        "filename": all_chunks[idx]["filename"],
        "text": all_chunks[idx]["text"],
        "distance": distances[0][i]
    } for i, idx in enumerate(indices[0])]

def generate_answer(query, retrieved_chunks):
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[Chunk {i+1} from {chunk['filename']}]\n{chunk['text']}\n\n"

    prompt = f"""You are a helpful assistant for Indecimal, a construction marketplace.
Answer the user's question using ONLY the context provided below.
If the answer is not found in the context, say "I don't have enough information to answer that."
Do not make up any information.

Context:
{context}

Question: {query}

Answer:"""

    # Fallback model list — tries each one until one works
    models = [
        "google/gemma-3-4b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "deepseek/deepseek-r1-distill-llama-8b:free",
    ]

    for model in models:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            result = response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
        except Exception:
            continue

    return "⚠️ All models are currently rate-limited. Please try again in a moment."


st.title("🏗️ Indecimal AI Assistant")
st.caption("Ask me anything about Indecimal's packages, policies, and services.")

with st.spinner("Loading knowledge base..."):
    embedder, index, all_chunks = build_rag_index()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question about Indecimal..."):
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks = retrieve_chunks(query, embedder, index, all_chunks)
            answer = generate_answer(query, chunks)

        st.markdown(answer)


        with st.expander("Retrieved Context"):
            for c in chunks:
                st.markdown(f"**Rank {c['rank']} | {c['filename']}**")
                st.caption(c['text'][:300])
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})