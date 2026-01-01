import os
import shutil
import json
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def copy_codes(save_directory):
    os.makedirs(save_directory, exist_ok=True)
    shutil.copy("./bridge/configuration_vaetki.py", f"{save_directory}/")
    shutil.copy("./bridge/modeling_vaetki.py", f"{save_directory}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        type=str,
    )
    parser.add_argument(
        "--prompts",
        metavar='N',
        type=str,
        nargs='+',
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--copy-codes",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if args.copy_codes:
        copy_codes(args.load)
    prompts = args.prompts
    if args.prompt_path is not None:
        prompts = []
        with open(args.prompt_path, encoding="utf8") as r:
            for line in r:
                prompts.append(json.loads(line)["text"])
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    model = AutoModelForCausalLM.from_pretrained(
        args.load,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")["input_ids"].to(model.device)
    start = time.time()
    outputs = model.generate(inputs, max_new_tokens=args.num_tokens_to_generate)
    print(tokenizer.batch_decode(outputs))
    print(time.time()-start)
