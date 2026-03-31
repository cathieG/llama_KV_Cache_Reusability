import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# This script:
# Loads all prompts, runs native 8B, and saves one result per prompt
# including timing + first-5-step logits for later KL analysis.

MODEL_ID = "meta-llama/Llama-3.1-8B"
MAX_NEW_TOKENS = 20
OUTPUT_FILE = "outputs/native_8b.jsonl"
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


PROMPTS = load_prompts(PROMPT_FILE)


def run_one_prompt(prompt, tokenizer, model):
    """
    Runs native 8B greedy decoding manually so we can:
    - measure TTFT
    - measure total latency
    - save first-5-step logits
    """
    device = model.device

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated_ids = input_ids.clone()
    current_input_ids = input_ids
    past = None

    step_logits_first5 = []

    # Timing
    t_start = time.perf_counter()
    first_decode_step_ms = None
    ttft_total_ms = None

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            step_start = time.perf_counter()

            if past is None:
                # First forward pass processes full prompt
                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True
                )
            else:
                # After first step, process only the newest token
                past_len = past.get_seq_length()
                attn_mask = torch.ones(
                    (1, past_len + current_input_ids.shape[1]),
                    dtype=attention_mask.dtype,
                    device=device
                )

                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=attn_mask,
                    past_key_values=past,
                    use_cache=True
                )

            logits = outputs.logits[:, -1, :]   # shape: [1, vocab_size]

            if step < LOGIT_STEPS_TO_SAVE:
                # Save raw logits for KL later
                step_logits_first5.append(
                    logits[0].detach().float().cpu().tolist()
                )

            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if step == 0:
                first_decode_step_ms = (time.perf_counter() - step_start) * 1000.0
                ttft_total_ms = (time.perf_counter() - t_start) * 1000.0

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_input_ids = next_token
            past = outputs.past_key_values

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    full_ids = generated_ids[0].tolist()
    prompt_len = input_ids.shape[1]
    gen_ids = full_ids[prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "generated_text": gen_text,
        "generated_token_ids": gen_ids,
        "prompt_token_count": prompt_len,
        "generation_token_count": len(gen_ids),
        "step_logits_first5": step_logits_first5,
        "first_decode_step_ms": first_decode_step_ms,
        "ttft_total_ms": ttft_total_ms,
        "total_latency_ms": total_latency_ms
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for item in PROMPTS:
            prompt_id = item["prompt_id"]
            prompt = item["prompt"]
            category = item["category"]
            length = item["length"]

            try:
                result = run_one_prompt(prompt, tokenizer, model)

                record = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "category": category,
                    "length": length,
                    "method": "native_8b",
                    "generated_text": result["generated_text"],
                    "generated_token_ids": result["generated_token_ids"],
                    "prompt_token_count": result["prompt_token_count"],
                    "generation_token_count": result["generation_token_count"],
                    "step_logits_first5": result["step_logits_first5"],
                    "first_decode_step_ms": result["first_decode_step_ms"],
                    "ttft_total_ms": result["ttft_total_ms"],
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
                    "method": "native_8b",
                    "generated_text": None,
                    "generated_token_ids": None,
                    "prompt_token_count": None,
                    "generation_token_count": None,
                    "step_logits_first5": None,
                    "first_decode_step_ms": None,
                    "ttft_total_ms": None,
                    "total_latency_ms": None,
                    "transfer_success": False,
                    "error": str(e)
                }

            f_out.write(json.dumps(record) + "\n")
            print(f"Finished prompt {prompt_id}", flush=True)


if __name__ == "__main__":
    main()
