#!/usr/bin/env python3

from typing import List
from llm import Llama, Dialog

MODEL_PATH = "/mnt/artemis/library/weights/meta/llama-2/7Bf"
TOKENIZER_PATH = "/mnt/artemis/library/weights/meta/llama-2/tokenizer.model"

def main():
    LLM = Llama.build(ckpt_dir=MODEL_PATH, tokenizer_path=TOKENIZER_PATH, max_seq_len=512, max_batch_size=8)

    dialogs: List[Dialog] = [
            [
            {"role": "system", "content": "Call: You are 'Jasmine', a superintelligent artificial intelligence support agent, your purpose is to assist the user with any request they have, returning short and precise answers in less than 100 words. You ONLY answer the question asked."},

            {"role": "user", "content": "How fast is an F16?"}
            ]
        ]

    results = LLM.chat_completion(dialogs, max_gen_len=None, temperature=0.6, top_p=0.9)

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        print("\n==================================\n")

if __name__ == "__main__":
    main()
