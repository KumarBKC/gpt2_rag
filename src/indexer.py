import os
import logging
from typing import List, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import AppConfig
from src.logger import get_logger

class DocumentIndexer:
    """Handles loading, splitting, and indexing of documents for RAG."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.get_default()
        self.logger = get_logger("Indexer")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)
            self.text_splitter = CharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE, 
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings or text splitter: {e}")
            raise

    def load_documents(self, folder_path: str) -> List:
        """
        Loads all .txt files from a folder.
        
        Args:
            folder_path (str): Path to the directory containing text documents.
            
        Returns:
            List: A list of LangChain Document objects.
        """
        all_documents = []
        if not os.path.exists(folder_path):
            self.logger.warning(f"Directory not found: {folder_path}")
            return []
            
        self.logger.info(f"Scanning directory: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                path = os.path.join(folder_path, filename)
                try:
                    loader = TextLoader(path, encoding='utf-8')
                    docs = loader.load()
                    all_documents.extend(docs)
                    self.logger.info(f"Loaded: {filename} ({len(docs)} segments)")
                except Exception as e:
                    self.logger.error(f"Error loading {filename}: {e}")
                    
        return all_documents

    def build_index(self, documents: List, save_path: str) -> bool:
        """
        Splits documents, creates embeddings, and saves the FAISS index.
        
        Args:
            documents (List): List of LangChain Document objects.
            save_path (str): Path where the FAISS index should be saved.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not documents:
            self.logger.error("No documents provided to index.")
            return False
            
        try:
            self.logger.info(f"Splitting {len(documents)} document(s) into chunks...")
            docs = self.text_splitter.split_documents(documents)
            
            self.logger.info("Creating embeddings and building FAISS index...")
            db = FAISS.from_documents(docs, self.embeddings)
            
            self.logger.info(f"Saving index to {save_path}...")
            db.save_local(save_path)
            self.logger.info("Indexing completed successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Failed to build or save index: {e}")
            return False

if __name__ == "__main__":
    # Internal test/example
    config = AppConfig.get_default()
    indexer = DocumentIndexer(config)
    raw_docs = indexer.load_documents(config.RAW_DATA_PATH)
    if raw_docs:
        indexer.build_index(raw_docs, config.INDEX_STORAGE_PATH)
