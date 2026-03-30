import os
import sys
from src.generator import RAGGenerator

def main():
    print("\n✅ RAG System Starting...")
    
    # Initialize generator with default relative paths
    try:
        rag = RAGGenerator(
            model_path="models/gpt2",
            index_path="data/index"
        )
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return

    print("\n✅ RAG System Ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break
        
        if not user_input.strip():
            continue
            
        print("Thinking...")
        response = rag.generate_answer(user_input)
        print(f"GPT-2: {response}\n")

if __name__ == "__main__":
    main()
