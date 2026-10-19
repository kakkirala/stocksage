from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import pandas as pd
import uuid

COLLECTION = "stock-market-v1"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Piece 1: load + build sentences
df = pd.read_csv('FINAL_FROM_DF.csv')
# df = df[df['TIMESTAMP'] >= '2017-01-01']
df = df[(df['TIMESTAMP'] >= '2017-01-01') & (df['TIMESTAMP'] < '2017-02-01')]
print(f"{len(df)} rows for 2017")

def row_to_sentence(r):
    return (f"On {r['TIMESTAMP']}, {r['SYMBOL']} opened at {r['OPEN']}, "
            f"high {r['HIGH']}, low {r['LOW']}, closed at {r['CLOSE']}, "
            f"volume {r['TOTTRDQTY']}.")

sentences = [row_to_sentence(r) for _, r in df.iterrows()]
print(f"Built {len(sentences)} sentences")

# Piece 2: embed + store
qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer(EMBED_MODEL)

qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vectors = embedder.encode(sentences, batch_size=64, show_progress_bar=True)

for i in range(0, len(sentences), 1000):
    batch = [
        PointStruct(id=str(uuid.uuid4()), vector=v.tolist(),
                    payload={"text": s})
        for s, v in zip(sentences[i:i+1000], vectors[i:i+1000])
    ]
    qdrant.upsert(collection_name=COLLECTION, points=batch)
    print(f"Stored {min(i+1000, len(sentences))}/{len(sentences)}")

print("Done — 2017 fully ingested")
