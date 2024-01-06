#!/usr/bin/env python3

from arch.llama.llm import Llama
from arch.utils import device_mapper, load_config, albedo, chat_playback, coder

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

MODEL_PATH = "/mnt/artemis/library/weights/meta/llama-2/7Bf"
TOKENIZER_PATH = "/mnt/artemis/library/weights/meta/llama-2/tokenizer.model"

B_INST, E_INST = "[INST]", "[/INST]" # IM_START / IM_END
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"

def main():
    LLM = Llama.build(ckpt_dir=MODEL_PATH, tokenizer_path=TOKENIZER_PATH, max_seq_len=512, max_batch_size=8, device=DEVICE)

    system = load_config('profiles.yml')['jasmine']
    print("Type 'exit()' to cancel.")
    print(f"{albedo(system, 'green')}")
    print("\n==================================\n") 

    log = [B_SYS] + [system] + [E_SYS]

    while True:
        x_inp = str(input("User: "))
        if x_inp == "exit()": break
        log += [B_INST] + [x_inp] + [E_INST]

        toks = [LLM.tokenizer.encode(' '.join(log), bos=True, eos=True)]
        new_toks, _ = LLM.generate(prompt_tokens=toks, max_gen_len=None, temperature=0.7, top_p=0.9, logprobs=False)
        response = [LLM.tokenizer.decode(t) for t in new_toks]
        
        # if logprobs:
        #     response = [{"generation": {"role": "assistant", "content": LLM.tokenizer.decode(t)}, "tokens": [LLM.tokenizer.decode(x) for x in t], "logprobs": logprobs_i,} for t, logprobs_i in zip(new_toks, logprobs)]
        # else:
        #     response = [LLM.tokenizer.decode(t) for t in new_toks]

        log += response
        print(albedo(log, 'red'))

        chat_playback(response[0])
        # coder(response[0])

        # url = 'https://images.newscientist.com/wp-content/uploads/2017/02/08190042/gettyimages-615305114.jpg'
        # raw_image = url2image(url)
        # response = V1.witness(raw_image)
        # print(response)

if __name__ == "__main__":
    main()