import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class DocumentIndexer:
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_documents(self, folder_path: str) -> List:
        """Loads all .txt files from a folder."""
        all_documents = []
        if not os.path.exists(folder_path):
            print(f"Directory not found: {folder_path}")
            return []
            
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                path = os.path.join(folder_path, filename)
                loader = TextLoader(path, encoding='utf-8')
                all_documents.extend(loader.load())
        return all_documents

    def build_index(self, documents: List, save_path: str):
        """Splits documents, creates embeddings, and saves the FAISS index."""
        if not documents:
            raise ValueError("No documents provided to index.")
            
        print(f"Splitting {len(documents)} document(s) into chunks...")
        docs = self.text_splitter.split_documents(documents)
        
        print("Creating embeddings and building FAISS index...")
        db = FAISS.from_documents(docs, self.embeddings)
        
        print(f"Saving index to {save_path}...")
        db.save_local(save_path)
        print("Success!")

if __name__ == "__main__":
    # Example usage
    indexer = DocumentIndexer()
    raw_docs = indexer.load_documents("data/raw")
    if raw_docs:
        indexer.build_index(raw_docs, "data/index")
    else:
        print("No documents found in data/raw")
