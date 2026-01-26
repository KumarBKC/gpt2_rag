# Local GPT-2 RAG: Your Private AI Knowledge Base

Welcome! This project is a lean, mean, local Retrieval-Augmented Generation (RAG) machine. It lets you chat with your own documents using a GPT-2 model running entirely on your own hardware. No APIs, no data leaving your machine—just pure, private AI.

## Why this project?

Most AI tools today rely on cloud APIs, which can be slow and raise privacy concerns. This setup demonstrates how you can:
- **Keep it Private**: Your data stays on your drive.
- **Run Offline**: Once set up, you don't even need an internet connection.
- **Own Your Knowledge**: Index your own `.txt` files and get instant, context-aware answers.

---

## What's Under the Hood?

- **Brain**: A local **GPT-2** model (`pytorch_model.bin`).
- **Memory**: **FAISS** (Facebook AI Similarity Search) for lightning-fast semantic search.
- **Translator**: **Sentence-Transformers** (`all-MiniLM-L6-v2`) to turn your text into meaningful numbers (embeddings).
- **Orchestrator**: **LangChain** to glue the retrieval and generation phases together.

---

## Getting Started

### 1. Prerequisites
Make sure you have **Python 3.8+** installed. You'll also need **PyTorch**—if you haven't installed it yet, head over to [pytorch.org](https://pytorch.org/) to get the right version for your system (CPU or GPU).

### 2. Setup the Environment
Clone this repository and grab the dependencies:

```bash
pip install langchain-community langchain-text-splitters faiss-cpu sentence-transformers transformers torch
```
> [!TIP]
> If you have an NVIDIA GPU, installing `faiss-gpu` instead of `faiss-cpu` will make indexing even faster!

### 3. Feed the AI
Put your knowledge in `doc1.txt` and `doc2.txt` in the root folder. The system will read these files to answer your questions.

### 4. Model Files
Check that you have these files in your folder (they make the GPT-2 engine run):
- `pytorch_model.bin`
- `config.json`
- `vocab.json`
- `merges.txt`

---

## How to Use It

### Step 1: Build the Index
First, "teach" the AI about your documents. This script splits your text into chunks and builds a searchable index in the `faiss_index/` folder.

```bash
python index.py
```

### Step 2: Verification (Optional)
If you want to make sure the model is loading correctly, run:
```bash
python test.py
```

### Step 3: Start Chatting!
The main event. Run the chat script to start a conversation with your documents.

```bash
python chat.py
```
Type your question, and GPT-2 will search your documents for the most relevant info before answering. Type `exit` to wrap up.

---

## Project Structure

- `chat.py`: The main entry point for the RAG chat loop.
- `index.py`: The tool used to ingest documents and build the FAISS index.
- `test.py`: A simple script to verify your local model works.
- `doc1.txt` / `doc2.txt`: Your knowledge base.
- `faiss_index/`: Where the system stores the "processed" version of your data.

---

## Troubleshooting

- **"Model not found"**: Double-check the path in `chat.py` and `test.py`. It's currently set to `C:/AI.ML WORK/gpt2_rag`. If you moved the folder, you'll need to update that line!
- **Slow Performance**: GPT-2 is relatively light, but if it feels sluggish, ensure you aren't running too many other heavy apps in the background.
- **Hallucinations**: GPT-2 is a smaller model. To keep it on track, we've set a low "temperature" in `chat.py`, but it works best with clear, concise documents.

---

Built with ❤️ for private, local-first AI.
