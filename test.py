from src.generator import RAGGenerator

def main():
    print("Testing local GPT-2 model loading...")
    
    try:
        # Load from the new modular system
        # We specify the paths relative to the root
        rag = RAGGenerator(
            model_path="models/gpt2",
            index_path="data/index"
        )
        print("✅ GPT-2 and FAISS loaded successfully!")
        
        # Simple test generation
        test_query = "Hello!"
        print(f"Test Query: {test_query}")
        print("Thinking...")
        response = rag.generate_answer(test_query)
        print(f"GPT-2 Response: {response}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
