import sys
from src.generator import RAGGenerator
from src.config import AppConfig
from src.logger import get_logger

def validate_system():
    logger = get_logger("Validator")
    logger.info(" Validating local model and vector store loading...")
    
    config = AppConfig.get_default()
    
    try:
        # Load generator components
        rag = RAGGenerator(config)
        logger.info(" Core components initialized successfully.")
        
        # Test context search
        test_query = "Who wrote the paper?"
        logger.info(f"Test Query: {test_query}")
        
        context = rag.search_context(test_query)
        if context:
            logger.info(" FAISS search returned valid context.")
        else:
            logger.warning(" FAISS search returned no results. Check if documents were indexed.")
            
        # Test full generation
        logger.info("Testing end-to-end generation...")
        response = rag.generate_answer(test_query)
        logger.info(f" Full Pipeline Response: {response}")
        
    except Exception as e:
        logger.error(f" System validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_system()
