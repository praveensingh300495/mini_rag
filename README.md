#  Mini RAG — Indecimal AI Assistant

A Retrieval-Augmented Generation (RAG) chatbot built for Indecimal, a construction marketplace. The assistant answers user questions strictly using internal company documents — no hallucinations, no general knowledge.

##  Live Demo
[Click here to view the deployed chatbot](https://minirag-praveen.streamlit.app/) <!-- Replace with your Streamlit Cloud URL -->

---

##  How It Works

This project implements a full RAG pipeline:

1. **Document Loading** — Internal markdown documents (policies, FAQs, package specs) are loaded from the `docs/` folder.
2. **Chunking** — Each document is split into overlapping chunks of ~100 words with a 20-word overlap to preserve context at boundaries.
3. **Embedding** — Each chunk is converted into a vector using `sentence-transformers` (`all-MiniLM-L6-v2`).
4. **Vector Indexing** — All chunk vectors are stored in a FAISS index for fast similarity search.
5. **Retrieval** — At query time, the user's question is embedded and the top-3 most semantically similar chunks are retrieved.
6. **Answer Generation** — The retrieved chunks are passed to an LLM via OpenRouter with a strict prompt: answer only from the provided context.
7. **Transparency** — The chatbot displays both the final answer and the retrieved source chunks.

---

##  Tech Stack

| Component | Tool | Reason |
|---|---|---|
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` | Free, fast, runs locally, no API key needed |
| Vector Store | `FAISS` (CPU) | Lightweight, local, no managed service needed |
| LLM | `google/gemma-3-4b-it:free` via OpenRouter | Free tier, no GPU required, good instruction following |
| Frontend | `Streamlit` | Fast to build, easy to deploy, supports chat UI |

---

##  Project Structure
```
mini-rag/
├── docs/
│   ├── doc1.md          # Indecimal Company Overview & Customer Journey
│   ├── doc2.md          # Package Comparison & Specification Wallets
│   └── doc3.md          # Customer Protection Policies & Quality System
├── rag_pipeline.ipynb   # Jupyter notebook — step-by-step RAG implementation
├── app.py               # Streamlit chatbot frontend
├── requirements.txt     # Python dependencies
├── .env                 # API key (not committed to git)
└── README.md
```

---

##  How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/mini-rag.git
cd mini-rag
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the root folder:
```
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxx
```

### 5. Run the chatbot
```bash
streamlit run app.py
```

---

##  Grounding & Hallucination Prevention

The LLM is explicitly prompted with:
> *"Answer the user's question using ONLY the context provided below. If the answer is not found in the context, say 'I don't have enough information to answer that.' Do not make up any information."*

This ensures the assistant never generates unsupported claims.

---

##  Example Queries

- "What are the construction packages and their prices?"
- "What is Indecimal's escrow payment system?"
- "How many quality checkpoints does Indecimal have?"
- "What does the maintenance program cover?"
- "What is the ceiling height across packages?"

---

##  Observations & Limitations

- Chunking by word count (100 words) works well for these documents but can occasionally split related pricing rows across chunks, affecting retrieval precision.
- The free LLM tier on OpenRouter may occasionally rate-limit — retrying with a different free model resolves this.
- FAISS L2 distance works well for semantic retrieval on this document size. For larger corpora, cosine similarity or HNSW indexing would be more efficient.

-The deployed app may show rate-limit errors on free OpenRouter tier during peak hours. The pipeline works correctly as demonstrated in the Jupyter notebook


