from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

question = "What was 3MINDIA's closing price on 6/28/2017?"
vector = embedder.encode(question).tolist()

hits = qdrant.query_points(
    collection_name="stock-market-v1",
    query=vector,
    limit=1,
).points

for h in hits:
    print(h.payload["text"], "| score:", round(h.score, 3))
