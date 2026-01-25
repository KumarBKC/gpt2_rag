import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Load the Knowledge Base (FAISS)
print("Loading Knowledge Base...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 2. Load the GPT-2 Model (Local)
print("Loading GPT-2 Model...")
model_path = "C:/AI.ML WORK/gpt2_rag"  # Pointing to root where model files exist
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

def ask_gpt2(question):
    # 1. SEARCH: Find the most relevant chunk
    docs = db.similarity_search(question, k=1)
    
    if not docs:
        return "I couldn't find any info in your documents."
    
    context_text = docs[0].page_content
    
    # 2. PROMPT: Stronger instructions for GPT-2
    # We force it to complete the sentence
    input_text = f"Keep your answer short and based ONLY on the text below.\n\nText: {context_text}\n\nQuestion: {question}\n\nAnswer:"
    
    # 3. GENERATE: Stricter settings
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,        # Increased slightly to allow full sentences
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.1,          # Low temp for factual answers
        top_k=40,
        repetition_penalty=1.2    # Penalize repetition
    )
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up: Just take the part after "Answer:"
    try:
        answer_only = full_response.split("Answer:")[1].strip()
    except IndexError:
        answer_only = full_response
        
    return answer_only

# 3. Start the Chat Loop
print("\n✅ RAG System Ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    print("Thinking...")
    response = ask_gpt2(user_input)
    print(f"GPT-2: {response}\n")
