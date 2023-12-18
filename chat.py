#!/usr/bin/env python3

import torch
from arch.llm import Llama
from arch.utils import load_config, chat_playback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(DEVICE).upper()}")

MODEL_DIR = "/mnt/artemis/library/weights/mistral/OpenHermes-2-Mistral-7B"

def main():
    LLM = Llama(MODEL_DIR, DEVICE)
    
    profile = load_config('profiles.yml')['jasmine']
    print(profile['call'])

    print("Type 'exit()' to cancel.")

    while True:
        x_inp = input("User: ")
        if x_inp == "exit()": break

        tok = LLM.encode(' '.join([profile['call'], x_inp]))
        new_tok = LLM.generate(tok, max_length=128, temp=1.0)
        startpos = len(tok[0])
        response = LLM.decode(new_tok[:,startpos:-1])

        chat_playback(response)

if __name__ == "__main__":
    main()