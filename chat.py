#!/usr/bin/env python3

from arch.utils import load_config

bos_token = "<s>"
eos_token = "</s>"

profile = load_config('profiles.yml')
agent = profile['james']

def main():
    print("Type 'exit()' to cancel.")
    
    hist = [bos_token, agent['call'], eos_token]
    while True:
        x_inp = input("User: ")
        if x_inp == "exit()":
            break
        hist.extend([bos_token, x_inp, eos_token])

        response = "ABC"
        print(f"Response: {response}")
        hist.extend([bos_token, response, eos_token])

if __name__ == "__main__":
    main()