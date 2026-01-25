from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Path to your GPT-2 folder
model_path = "C:/AI.ML WORK/gpt2_rag"

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

print("GPT-2 loaded successfully!")

# Test generation
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
