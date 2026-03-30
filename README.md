# Local GPT-2 RAG: Your Private AI Knowledge Base

This project is a local Retrieval-Augmented Generation (RAG) system using GPT-2 and FAISS. It allows you to chat with your own documents privately and offline.

## Project Structure

The project has been reorganized into a modular structure:

- `data/`
  - `raw/`: Place your `.txt` documents here.
  - `index/`: Stores the generated FAISS index.
- `models/`
  - `gpt2/`: Contains local GPT-2 model weights (`pytorch_model.bin`, `config.json`, etc.).
- `src/`: Core logic modules.
  - `indexer.py`: Logic for document processing and indexing.
  - `generator.py`: Logic for RAG and GPT-2 inference.
- `chat.py`: Main interactive chat entry point.
- `index.py`: Script to build/update the knowledge base.
- `test.py`: Script to verify model and index loading.
- `requirements.txt`: Python dependencies.

## How to Run

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your text files (e.g., `doc1.txt`, `doc2.txt`) in the `data/raw/` directory.

### 3. Build the Index
To process your documents and create the searchable knowledge base, run:
```bash
python index.py
```
This will create/update the index in `data/index/`.

### 4. Verify the Setup (Optional)
Run the test script to ensure the model and index are loading correctly:
```bash
python test.py
```

### 5. Start Chatting
Launch the interactive chat loop:
```bash
python chat.py
```
Type your questions and GPT-2 will answer using the documents in your knowledge base. Type `exit` to quit.

## Troubleshooting
- **Missing Model**: Ensure the GPT-2 files are in `models/gpt2/`.
- **No Documents**: Ensure your `.txt` files are in `data/raw/`.
- **Absolute Paths**: This version uses relative paths, so always run the scripts from the project root.
