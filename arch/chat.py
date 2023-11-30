#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("LLaMA-model-name")
# model = AutoModelForCausalLM.from_pretrained("LLaMA-model-name")

MODEL_DIR = "/mnt/artemis/library/weights/meta/llama-2/7Bf/hf/"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

def encode(prompt):
    return tokenizer.encode(prompt, return_tensors="pt")

def generate(emb):
    with torch.no_grad():
        logits = model(input_ids).logits
    return logits

def decode(logits):
    tok = torch.argmax(logits[0, -1, :]).item()
    return tokenizer.decode(tok)

user = "The cat sat on the "

input_ids = encode(user)
logits = generate(input_ids)
tok = decode(logits)

print(tok)

# def predict_next_word(text):
#     input_ids = tokenizer.encode(text, return_tensors="pt")

#     # Get the model's predictions
#     with torch.no_grad():
#         logits = model(input_ids).logits

#     # Find the most likely next token
#     predicted_token_id = torch.argmax(logits[0, -1, :]).item()
#     predicted_word = tokenizer.decode([predicted_token_id])

#     return predicted_word

# # Example usage
# input_text = "The quick brown fox jumps over the lazy"
# next_word = predict_next_word(input_text)
# print(f"Next word prediction: {next_word}")