from flask import Flask, request, render_template
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# File paths
PDF_PATH = "ipc.pdf"
INDEX_FILE = "ipc_index.faiss"
CHUNKS_FILE = "ipc_chunks.pkl"

# Load models
print("🔹 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🔹 Loading TinyLlama model...")
tokenizer = AutoTokenizer.from_pretrained("PY007/TinyLlama-1.1B-Chat-v0.1")
model = AutoModelForCausalLM.from_pretrained(
    "PY007/TinyLlama-1.1B-Chat-v0.1",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load or create FAISS index
def load_pdf_chunks(pdf_path, max_chunk_length=500):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts = text.split('\n')
                chunk = ''
                for part in parts:
                    if len(chunk) + len(part) < max_chunk_length:
                        chunk += part + ' '
                    else:
                        chunks.append(chunk.strip())
                        chunk = part + ' '
                if chunk:
                    chunks.append(chunk.strip())
    return chunks

if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    print("✅ Loading FAISS index and chunks...")
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, 'rb') as f:
        chunks = pickle.load(f)
else:
    print("📄 Creating FAISS index from IPC PDF...")
    chunks = load_pdf_chunks(PDF_PATH)
    chunk_embeddings = embedder.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(np.array(chunk_embeddings))
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, 'wb') as f:
        pickle.dump(chunks, f)

# Helper: Get context from FAISS
def get_relevant_context(query, k=5):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return "\n".join([chunks[i] for i in I[0]])

# Helper: Generate response
def generate_answer(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Renamed route function to avoid 'index' name conflict
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    query = request.form["query"]
    context = get_relevant_context(query)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generate_answer(prompt)
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
