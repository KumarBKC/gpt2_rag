import os
import torch
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class RAGGenerator:
    def __init__(
        self,
        model_path: str = "models/gpt2",
        index_path: str = "data/index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        print(f"Initializing RAG system with model from {model_path}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        if not os.path.exists(index_path):
            print(f"Warning: Index path {index_path} not found. Please run indexer first.")
            self.db = None
        else:
            self.db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            
        print("Loading local GPT-2 model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def search_context(self, query: str, k: int = 1) -> Optional[str]:
        """Searches the FAISS index for relevant context."""
        if not self.db:
            return None
        
        docs = self.db.similarity_search(query, k=k)
        if not docs:
            return None
        
        return docs[0].page_content

    def generate_answer(self, question: str, max_new_tokens: int = 50) -> str:
        """Retrieves context and generates an answer using GPT-2."""
        context = self.search_context(question)
        
        if not context:
            return "I couldn't find any info in your documents."

        # Construct prompt
        prompt = (
            f"Keep your answer short and based ONLY on the text below.\n\n"
            f"Text: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1,
                top_k=40,
                repetition_penalty=1.2
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the answer part
        if "Answer:" in full_response:
            return full_response.split("Answer:")[1].strip()
        
        return full_response.strip()

if __name__ == "__main__":
    # Quick test
    rag = RAGGenerator()
    answer = rag.generate_answer("Hello?")
    print(f"GPT-2: {answer}")
