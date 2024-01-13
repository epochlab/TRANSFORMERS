#!/usr/bin/env python3

from typing import List, Literal, TypedDict

from arch.llama.llm import Llama
from arch.utils import device_mapper, load_config, chat_playback, coder, albedo

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

MODEL_PATH = "/mnt/artemis/library/weights/meta/llama-2/7Bf"
# MODEL_PATH = "/mnt/artemis/library/weights/meta/llama-2/7Bf"
TOKENIZER_PATH = "/mnt/artemis/library/weights/meta/llama-2/tokenizer.model"

class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

Dialog = List[Message]
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"

def main():
    LLM = Llama.build(ckpt_dir=MODEL_PATH, tokenizer_path=TOKENIZER_PATH, max_seq_len=512, max_batch_size=8, device=DEVICE)

    system_call = load_config('profiles.yml')['quentin']
    log: List[Dialog] = [[{"role": "system", "content": system_call}]]

    print("Type 'exit()' to cancel.")
    print(f"{albedo(system_call, 'green')}")
    print("\n==================================\n") 

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit()": break
        log[-1].append({"role": "user", "content": user_input})

        dialog = log[-1]
        if dialog[0]["role"] == "system":
            dialog = [{"role": dialog[1]["role"], "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"]}] + dialog[2:]

        dialog_toks: List[int] = sum([LLM.tokenizer.encode(f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()}", bos=True, eos=True,) for prompt, answer in zip(dialog[::2], dialog[1::2],)],[],)
        dialog_toks += LLM.tokenizer.encode(f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}", bos=True, eos=False)
        toks = [dialog_toks]

        logprobs = False
        new_toks, tok_logprobs = LLM.generate(prompt_tokens=toks, max_gen_len=None, temperature=0.6, top_p=0.9, logprobs=logprobs)

        if logprobs:
            result = [{"generation": {"role": "assistant", "content": LLM.tokenizer.decode(t)}, "tokens": [LLM.tokenizer.decode(x) for x in t], "logprobs": logprobs_i,} for t, logprobs_i in zip(new_toks[0], tok_logprobs)]
        else:
            result = {"generation": {"role": "assistant", "content": LLM.tokenizer.decode(new_toks[0])}}

        log[-1].append(result['generation'])
        # print(albedo(dialog, "red")) # Print chat history

        chat_playback(f"> {result['generation']['content']}")
        coder(result['generation']['content'])

if __name__ == "__main__":
    main()
