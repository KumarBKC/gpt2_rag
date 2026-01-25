import os
# IMPORTS FOR YOUR VERSION
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Define where your documents are
documents_paths = ["doc1.txt", "doc2.txt"]

all_documents = []

# 2. Load the text from the files
print("Loading documents...")
for path in documents_paths:
    if os.path.exists(path):
        loader = TextLoader(path, encoding='utf-8')
        all_documents.extend(loader.load())
    else:
        print(f"Warning: {path} not found.")

if not all_documents:
    print("No documents found! Please check doc1.txt and doc2.txt")
    exit()

# 3. Split text into smaller chunks
# We use the new CharacterTextSplitter imported above
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(all_documents)

# 4. Create Embeddings
print("Creating embeddings (this might take a moment)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Create and Save the FAISS Index
print("Building FAISS index...")
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print("Success! Index created and saved in 'faiss_index' folder.")