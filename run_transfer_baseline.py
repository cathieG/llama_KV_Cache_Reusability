import json
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

MODEL_1B = "meta-llama/Llama-3.2-1B"
MODEL_8B = "meta-llama/Llama-3.1-8B"
MAX_NEW_TOKENS = 20
PROMPT_FILE = "prompts/prompts.jsonl"
LOGIT_STEPS_TO_SAVE = 5


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def expand_kv_dim(x: torch.Tensor, method: str) -> torch.Tensor:
    if method == "duplication":
        return torch.cat([x, x], dim=-1)
    elif method == "zero_padding":
        pad = torch.zeros_like(x)
        return torch.cat([x, pad], dim=-1)
    else:
        raise ValueError(f"Unknown method: {method}")


def transform_1b_to_8b_pkv(past_key_values_1b, method: str):
    transformed = []

    for layer_idx in range(len(past_key_values_1b)):
        k, v = past_key_values_1b[layer_idx]

        k2 = expand_kv_dim(k, method)
        v2 = expand_kv_dim(v, method)

        # 1B has 16 layers, 8B has 32 layers, so duplicate each layer twice
        transformed.append((k2, v2))
        transformed.append((k2.clone(), v2.clone()))

    return tuple(transformed)


def run_one_prompt(prompt, tokenizer, model_1b, model_8b, method, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if inputs["input_ids"].shape[1] < 1:
        raise ValueError("Prompt tokenization produced empty input.")

    prompt_len = inputs["input_ids"].shape[1]

    # ---------------------------
    # 1B PREFILL
    # ---------------------------
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        out_1b = model_1b(**inputs, use_cache=True)
    prefill_1b_ms = (time.perf_counter() - t_prefill_start) * 1000.0

    pkv_1b = out_1b.past_key_values

    # ---------------------------
    # TRANSFORM KV
    # ---------------------------
    t_transform_start = time.perf_counter()
    pkv_8b_tuple = transform_1b_to_8b_pkv(pkv_1b, method)
    pkv_8b = DynamicCache.from_legacy_cache(pkv_8b_tuple)
    transform_ms = (time.perf_counter() - t_transform_start) * 1000.0

    # ---------------------------
    # 8B MANUAL GREEDY DECODE
    # ---------------------------
    current_input_ids = inputs["input_ids"][:, -1:]
    generated_ids = inputs["input_ids"].clone()
    past = pkv_8b

    step_logits_first5 = []

    decode_start = time.perf_counter()
    resume_ttft_ms = None

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            step_start = time.perf_counter()

            past_len = past.get_seq_length()
            attn_mask = torch.ones(
                (1, past_len + current_input_ids.shape[1]),
                dtype=inputs["attention_mask"].dtype,
                device=device
            )

            outputs = model_8b(
                input_ids=current_input_ids,
                attention_mask=attn_mask,
                past_key_values=past,
                use_cache=True
            )

            logits = outputs.logits[:, -1, :]  # shape: [1, vocab_size]

            if step < LOGIT_STEPS_TO_SAVE:
                step_logits_first5.append(
                    logits[0].detach().float().cpu().tolist()
                )

            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if step == 0:
                resume_ttft_ms = (time.perf_counter() - step_start) * 1000.0

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_input_ids = next_token
            past = outputs.past_key_values

    decode_8b_ms = (time.perf_counter() - decode_start) * 1000.0
    ttft_total_ms = prefill_1b_ms + transform_ms + resume_ttft_ms
    total_latency_ms = prefill_1b_ms + transform_ms + decode_8b_ms

    full_ids = generated_ids[0].tolist()
    gen_ids = full_ids[prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "generated_text": gen_text,
        "generated_token_ids": gen_ids,
        "prompt_token_count": prompt_len,
        "generation_token_count": len(gen_ids),
        "step_logits_first5": step_logits_first5,
        "resume_ttft_ms": resume_ttft_ms,
        "ttft_total_ms": ttft_total_ms,
        "prefill_1b_ms": prefill_1b_ms,
        "transform_ms": transform_ms,
        "decode_8b_ms": decode_8b_ms,
        "total_latency_ms": total_latency_ms
    }


def main(method):
    output_file = f"outputs/transfer_{method}.jsonl"
    prompts = load_prompts(PROMPT_FILE)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_8B)

    print("Loading 1B model...", flush=True)
    model_1b = AutoModelForCausalLM.from_pretrained(
        MODEL_1B,
        dtype=torch.float16,
        device_map="auto"
    )
    model_1b.eval()

    print("Loading 8B model...", flush=True)
    model_8b = AutoModelForCausalLM.from_pretrained(
        MODEL_8B,
        dtype=torch.float16,
        device_map="auto"
    )
    model_8b.eval()

    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in prompts:
            prompt_id = item["prompt_id"]
            prompt = item["prompt"]
            category = item["category"]
            length = item["length"]

            try:
                result = run_one_prompt(
                    prompt, tokenizer, model_1b, model_8b, method, device
                )

                record = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "category": category,
                    "length": length,
                    "method": method,
                    "generated_text": result["generated_text"],
                    "generated_token_ids": result["generated_token_ids"],
                    "prompt_token_count": result["prompt_token_count"],
                    "generation_token_count": result["generation_token_count"],
                    "step_logits_first5": result["step_logits_first5"],
                    "resume_ttft_ms": result["resume_ttft_ms"],
                    "ttft_total_ms": result["ttft_total_ms"],
                    "prefill_1b_ms": result["prefill_1b_ms"],
                    "transform_ms": result["transform_ms"],
                    "decode_8b_ms": result["decode_8b_ms"],
                    "total_latency_ms": result["total_latency_ms"],
                    "transfer_success": True,
                    "error": None
                }

            except Exception as e:
                record = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "category": category,
                    "length": length,
                    "method": method,
                    "generated_text": None,
                    "generated_token_ids": None,
                    "prompt_token_count": None,
                    "generation_token_count": None,
                    "step_logits_first5": None,
                    "resume_ttft_ms": None,
                    "ttft_total_ms": None,
                    "prefill_1b_ms": None,
                    "transform_ms": None,
                    "decode_8b_ms": None,
                    "total_latency_ms": None,
                    "transfer_success": False,
                    "error": str(e)
                }

            f_out.write(json.dumps(record) + "\n")
            print(f"{method}: finished prompt {prompt_id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["zero_padding", "duplication"])
    args = parser.parse_args()
    main(args.method)
