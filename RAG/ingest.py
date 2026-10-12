import os
import chromadb


def ingest_knowledge_base():
    # Determine directory path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kb_file_path = os.path.join(base_dir, "knowledge_base.txt")
    db_folder_path = os.path.join(base_dir, "chroma_db")

    # 1. Read the knowledge base file
    if not os.path.exists(kb_file_path):
        raise FileNotFoundError(f"Could not find {kb_file_path}")

    with open(kb_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. Split text into individual exercise chunks
    # We split on double newlines or horizontal rule dividers ('---')
    raw_chunks = content.split("---")
    chunks = [c.strip() for c in raw_chunks if c.strip()]

    # 3. Initialize local persistent ChromaDB client
    client = chromadb.PersistentClient(path=db_folder_path)

    # Reset/Create the collection
    collection_name = "pilates_kb"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection didn't exist yet

    collection = client.create_collection(name=collection_name)

    # 4. Ingest documents into vector store
    documents = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({"source": "knowledge_base.txt"})
        ids.append(f"exercise_{idx + 1}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Success! Ingested {len(documents)} chunks into ChromaDB directory: '{db_folder_path}'.")


if __name__ == "__main__":
    ingest_knowledge_base()