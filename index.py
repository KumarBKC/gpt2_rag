from src.indexer import DocumentIndexer
from src.config import AppConfig
from src.logger import get_logger

def run_indexing():
    logger = get_logger("IndexerApp")
    logger.info(" Starting Document Indexing pipeline...")
    
    config = AppConfig.get_default()
    indexer = DocumentIndexer(config)
    
    # 1. Load documents from raw data folder
    raw_docs = indexer.load_documents(config.RAW_DATA_PATH)
    
    if not raw_docs:
        logger.error(f"No documents found in '{config.RAW_DATA_PATH}'. Please add .txt files.")
        return

    # 2. Build and save the index
    success = indexer.build_index(raw_docs, config.INDEX_STORAGE_PATH)
    
    if success:
        logger.info(f"Success! Index built and stored in '{config.INDEX_STORAGE_PATH}'.")
    else:
        logger.error(" Indexing pipeline failed.")

if __name__ == "__main__":
    run_indexing()