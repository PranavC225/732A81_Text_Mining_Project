import json, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CLEAN_DIR  = "data/clean"
CHROMA_DIR = "data/chroma"

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80

print("Loading embedding model...")
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)

vectorstore = Chroma(
    collection_name="liu_housing",
    embedding_function=embedder,
    persist_directory=CHROMA_DIR,
)

all_chunks, all_metas = [], []

for fname in sorted(os.listdir(CLEAN_DIR)):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(CLEAN_DIR, fname), encoding="utf-8") as f:
        doc = json.load(f)

    chunks = splitter.split_text(doc["text"])
    metas  = [{"source": doc["name"], "url": doc["url"]} for _ in chunks]
    all_chunks.extend(chunks)
    all_metas.extend(metas)
    print(f"  {doc['name']}: {len(chunks)} chunks")

print(f"\nIndexing {len(all_chunks)} chunks into ChromaDB...")
vectorstore.add_texts(texts=all_chunks, metadatas=all_metas)
print("Done.")