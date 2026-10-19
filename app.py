from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama

app = FastAPI()
qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

class AskRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/ask")
def ask(req: AskRequest):
    vector = embedder.encode(req.question).tolist()
    hits = qdrant.query_points(
        collection_name="stock-market-v1", query=vector, limit=3
    ).points
    context = "\n".join(h.payload["text"] for h in hits)
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Each context line looks like: On DATE, SYMBOL opened at X, high Y, low Z, closed at PRICE, volume V. Find the line matching the asked symbol and date, and report its numbers briefly. Answer using ONLY the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"},
        ],
    )
    return {"answer": response["message"]["content"], "context": context}
