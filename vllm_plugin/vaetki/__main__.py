if __name__ == "__main__":
    import os
    import random
    import string
    import json
    from vllm import LLM, SamplingParams

    random.seed(42)
    llm = LLM(
        model=os.getenv("CKPT_PATH"),
        trust_remote_code=True,
        pipeline_parallel_size=8,
        additional_config={"precision_level": 2},   # 0-2; 2 is most precise
        # enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.,
        max_tokens=1024,
        # ignore_eos=True,
    )

    prompt_path = os.getenv("PROMPT_PATH", None)
    if prompt_path is not None:
        with open(prompt_path, encoding="utf8") as r:
            line = r.readline()
        prompt = json.loads(line)["text"]
    else:
        prompt = os.getenv("PROMPT")

    # Warmup steps
    for _ in range(int(os.getenv("NUM_WARMUP", 2))):
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
        llm.generate([word], sampling_params)
    
    # Actual step
    output = llm.generate([prompt], sampling_params)
    print(f"Input token length: {len(llm.get_tokenizer()(prompt)['input_ids'])}")
    print(f"Generated text: {output[0].outputs[0].text!r}")
