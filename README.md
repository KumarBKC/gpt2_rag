# GPT-2 RAG Project

This project implements a Retrieval-Augmented Generation (RAG) workflow using a local GPT-2 model and FAISS for vector search. It demonstrates how to ingest custom text documents, create a semantic index, and load a local Large Language Model (LLM).

## Features

- **Document Ingestion**: Loads text from `doc1.txt` and `doc2.txt`.
- **Vector Indexing**: Uses `LangChain` and `FAISS` to create a searchable vector store of your documents.
- **Embeddings**: Utilizes `sentence-transformers/all-MiniLM-L6-v2` for high-quality text embeddings.
- **Local Model**: Loads and runs a fine-tuned or standard GPT-2 model locally without external API dependencies.

## Prerequisites

- Python 3.8 or higher
- [PyTorch](https://pytorch.org/) (configured for your system)

## Installation

1.  **Clone or set up the repository**.

2.  **Install dependencies**:
    Run the following command to install the required Python packages:

    ```bash
    pip install langchain-community langchain-text-splitters faiss-cpu sentence-transformers transformers torch
    ```

    *Note: If you have a GPU, you may want to install `faiss-gpu` instead of `faiss-cpu`.*

3.  **Model Setup**:
    Ensure the GPT-2 model files are present in the root directory:
    - `pytorch_model.bin`
    - `config.json`
    - `vocab.json`
    - `merges.txt`

4.  **Data Setup**:
    Add your content to `doc1.txt` and `doc2.txt` in the root directory.

## Usage

### 1. Create the Index

Run `index.py` to process your documents and build the FAISS vector index. This script will split your text, generate embeddings, and save the index to the `faiss_index` folder.

```bash
python index.py
```

**Output**:
- Creates a `faiss_index/` directory containing the searchable index.

### 2. Test the Model

Run `test.py` to verify that your local GPT-2 model loads correctly and can generate text.

```bash
python test.py
```

**Output**:
- Prints "GPT-2 loaded successfully!" and a sample text generation result.

## Project Structure

- `index.py`: Script to load documents and save the FAISS index.
- `test.py`: Script to load the local GPT-2 model and run inference.
- `config.json` / `pytorch_model.bin`: Local model files.
- `faiss_index/`: (Generated) Stores the vector database.

## Troubleshooting

- **Memory Issues**: If you run into memory errors during indexing, try creating smaller chunks in `index.py` by reducing `chunk_size` in `CharacterTextSplitter`.
- **Model Not Found**: Ensure the path in `test.py` (`model_path = "C:/AI.ML WORK/gpt2_rag"`) matches your actual directory structure.
