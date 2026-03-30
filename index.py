from src.indexer import DocumentIndexer

def main():
    print("🚀 Starting Document Indexing...")
    
    indexer = DocumentIndexer()
    
    # Load from default data/raw folder
    raw_docs = indexer.load_documents("data/raw")
    
    if not raw_docs:
        print("❌ No .txt documents found in 'data/raw/'. Please add files first.")
        return

    # Build and save index to data/index folder
    try:
        indexer.build_index(raw_docs, "data/index")
        print("\n✅ Successfully built and saved FAISS index to 'data/index/'.")
    except Exception as e:
        print(f"❌ Error building index: {e}")

if __name__ == "__main__":
    main()