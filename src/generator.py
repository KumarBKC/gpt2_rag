import os
import torch
import logging
from typing import List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.config import AppConfig
from src.logger import get_logger

class RAGGenerator:
    """Retrieves context from FAISS and generates answers using a local GPT-2 model."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.get_default()
        self.logger = get_logger("Generator")
        
        self.logger.info("Initializing RAG system components...")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)
            self._load_vector_db()
            self._load_llm()
        except Exception as e:
            self.logger.critical(f"Failed to initialize RAGGenerator: {e}")
            raise

    def _load_vector_db(self):
        """Loads the FAISS index if it exists."""
        if not os.path.exists(self.config.INDEX_STORAGE_PATH):
            self.logger.warning(
                f"Index path {self.config.INDEX_STORAGE_PATH} not found. "
                "Search will be disabled until an index is built."
            )
            self.db = None
        else:
            try:
                self.db = FAISS.load_local(
                    self.config.INDEX_STORAGE_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                self.logger.info("FAISS vector store loaded successfully.")
            except Exception as e:
                self.logger.error(f"Error loading FAISS index: {e}")
                self.db = None

    def _load_llm(self):
        """Loads the local GPT-2 model and tokenizer."""
        model_path = self.config.MODEL_PATH
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
            
        try:
            self.logger.info(f"Loading local GPT-2 model from {model_path}...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info("LLM loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading LLM: {e}")
            raise

    def search_context(self, query: str, k: int = 1) -> Optional[str]:
        """
        Searches the FAISS index for relevant context.
        
        Args:
            query (str): The user's question or search query.
            k (int): Number of documents to retrieve.
            
        Returns:
            Optional[str]: The content of the most relevant document chunk.
        """
        if not self.db:
            self.logger.error("Vector store not available.")
            return None
        
        try:
            docs = self.db.similarity_search(query, k=k)
            if not docs:
                return None
            return docs[0].page_content
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return None

    def generate_answer(self, question: str) -> str:
        """
        Retrieves context and generates an answer using GPT-2.
        
        Args:
            question (str): The user's query.
            
        Returns:
            str: The generated response.
        """
        context = self.search_context(question)
        
        if not context:
            return "I couldn't find any relevant information in the documents."

        self.logger.info(f"Generating answer for: {question[:50]}...")
        
        # Build a structured prompt
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer correctly based on the context above. Keep it concise.\n"
            f"Answer:"
        )

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=self.config.TEMPERATURE,
                    top_k=self.config.TOP_K,
                    repetition_penalty=self.config.REPETITION_PENALTY
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Use a more reliable way to extract the answer
            if "Answer:" in full_response:
                return full_response.split("Answer:")[1].strip()
            
            self.logger.warning("Prompt delimiter 'Answer:' not found in LLM output.")
            return full_response.strip()
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return f"Error during generation: {e}"

if __name__ == "__main__":
    # Internal test/example
    rag = RAGGenerator()
    resp = rag.generate_answer("Who is the author?")
    print(f"Response: {resp}")
