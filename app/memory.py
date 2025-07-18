# memory.py
import chromadb
from sentence_transformers import SentenceTransformer

# Setup Chroma client and collection
client = chromadb.Client()
collection = client.get_or_create_collection(name="image_captions")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # fast & lightweight

def store_caption(image_id, caption):
    embedding = model.encode(caption).tolist()
    collection.add(
        documents=[caption],
        embeddings=[embedding],
        ids=[image_id]
    )

def query_caption(query_text, top_k=3):
    query_embedding = model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"]
