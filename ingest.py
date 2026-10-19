from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

COLLECTION = "stock-market-v1"
EMBED_MODEL = "all-MiniLM-L6-v2"   # <-- check this matches your Ragscope value first
def main():
    qdrant = QdrantClient(host="localhost", port=6333)
    embedder = SentenceTransformer(EMBED_MODEL)

    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    sentences = [
        # paste your 4 sentences here, one per line in quotes
    "On 6/28/2017, 20MICRONS opened at 37.8, high 37.8, low 36.15, closed at 36.85, volume 27130.",
    "On 6/28/2017, 3MINDIA opened at 13425.15, high 13469.55, low 12920, closed at 13266.7, volume 2290.",
    "On 6/28/2017, 3IINFOTECH opened at 4.1, high 4.85, low 4, closed at 4.55, volume 20157058.",
    "On 6/28/2017, 63MOONS opened at 61, high 61.9, low 60.35, closed at 61, volume 27701.",
    ]
    points = []
    for s in sentences:
        vector = embedder.encode(s).tolist()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": s},
            )
        )

    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"Ingested {len(points)} sentences")


if __name__ == "__main__":
    main()

