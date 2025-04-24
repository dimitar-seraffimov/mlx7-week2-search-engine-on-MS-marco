from chromadb import PersistentClient
from chromadb.config import Settings

# CONFIG
CHROMA_DB_DIR = "../chromadb"
CHROMA_COLLECTION_NAME = "document"

# Connect to ChromaDB (correct modern way)
chroma_client = PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_DB_DIR,
        anonymized_telemetry=False,
    )
)

# Load collection
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"distance_metric": "cosine"},
    embedding_function=None,  # <-- critical for custom embeddings
)

# Display sample data
print("\n🔍 Peeking into ChromaDB collection:")
peek = collection.peek()
for i, doc in enumerate(peek["documents"]):
    print(f"\n[Doc {i}]")
    print("📄 Text:", doc)
    print("🔢 ID:", peek["ids"][i])
    print("📐 Embedding (dim):", len(peek["embeddings"][i]))

print("\n📊 [DEBUG] Total docs in collection:", collection.count())
