import sys
from src.generator import RAGGenerator
from src.config import AppConfig
from src.logger import get_logger

def main():
    logger = get_logger("ChatApp")
    logger.info("Initializing GPT-2 RAG system...")
    
    config = AppConfig.get_default()
    
    try:
        rag = RAGGenerator(config)
    except Exception as e:
        logger.critical(f"Starting Chat System failed: {e}")
        sys.exit(1)

    print(" RAG System Ready!")
    print("Commands: 'exit' to quit, 'help' for info.")

    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == "exit":
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == "help":
                print("Type your question to search the knowledge base using local GPT-2.")
                continue
                
            print("Thinking...")
            response = rag.generate_answer(user_input)
            print(f"GPT-2: {response}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Interaction error: {e}")

    print("\n Goodbye!")

if __name__ == "__main__":
    main()
