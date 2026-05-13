import os, json, time, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from adapter_common import TargetLayerAwareKVAdapter

MODEL_1B = "meta-llama/Llama-3.2-1B"
MODEL_8B = "meta-llama/Llama-3.1-8B"
MAX_NEW_TOKENS = 20
PROMPT_FILE = "prompts/prompts.jsonl"
LOGIT_STEPS_TO_SAVE = 5


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_adapter(path, device, dtype):
    ckpt = torch.load(path, map_location="cpu")
    md = ckpt["metadata"]
    adapter = TargetLayerAwareKVAdapter(
        num_target_layers=md.get("num_target_layers", 32),
        input_dim=md.get("input_dim", 64), output_dim=md.get("output_dim", 128),
        hidden_dim=md.get("hidden_dim", 256))
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    return adapter.to(device=device, dtype=dtype).eval(), md


def transform_pkv(pkv_1b, adapter, device, adapter_dtype, model_dtype):
    transformed = []
    with torch.no_grad():
        for src_i, (k, v) in enumerate(pkv_1b):
            k = k.to(device=device, dtype=adapter_dtype)
            v = v.to(device=device, dtype=adapter_dtype)
            for tgt_i in (2 * src_i, 2 * src_i + 1):
                k2, v2 = adapter.forward_target_layer(tgt_i, k, v)
                transformed.append((k2.to(dtype=model_dtype), v2.to(dtype=model_dtype)))
    return DynamicCache.from_legacy_cache(tuple(transformed))


def run_one(prompt, tokenizer, model_1b, model_8b, adapter, device, adapter_dtype, model_dtype):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    if prompt_len < 1:
        raise ValueError("Prompt tokenization produced empty input.")

    t0 = time.perf_counter()
    with torch.no_grad():
        out_1b = model_1b(**inputs, use_cache=True)
    prefill_1b_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    past = transform_pkv(out_1b.past_key_values, adapter, device, adapter_dtype, model_dtype)
    transform_ms = (time.perf_counter() - t0) * 1000

    cur = inputs["input_ids"][:, -1:]
    generated = inputs["input_ids"].clone()
    step_logits_first5 = []
    decode_start = time.perf_counter(); resume_ttft_ms = None

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            step_start = time.perf_counter()
            past_len = past.get_seq_length()
            attn_mask = torch.ones((1, past_len + cur.shape[1]), dtype=inputs["attention_mask"].dtype, device=device)
            out = model_8b(input_ids=cur, attention_mask=attn_mask, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            if step < LOGIT_STEPS_TO_SAVE:
                step_logits_first5.append(logits[0].detach().float().cpu().tolist())
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            if step == 0:
                resume_ttft_ms = (time.perf_counter() - step_start) * 1000
            generated = torch.cat([generated, nxt], dim=1)
            cur, past = nxt, out.past_key_values

    decode_8b_ms = (time.perf_counter() - decode_start) * 1000
    gen_ids = generated[0].tolist()[prompt_len:]
    return {
        "generated_text": tokenizer.decode(gen_ids, skip_special_tokens=True),
        "generated_token_ids": gen_ids,
        "prompt_token_count": prompt_len,
        "generation_token_count": len(gen_ids),
        "step_logits_first5": step_logits_first5,
        "resume_ttft_ms": resume_ttft_ms,
        "ttft_total_ms": prefill_1b_ms + transform_ms + resume_ttft_ms,
        "prefill_1b_ms": prefill_1b_ms,
        "transform_ms": transform_ms,
        "decode_8b_ms": decode_8b_ms,
        "total_latency_ms": prefill_1b_ms + transform_ms + decode_8b_ms,
    }


def main(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    prompts = load_prompts(args.prompt_file)
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    adapter_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_8B)
    model_1b = AutoModelForCausalLM.from_pretrained(MODEL_1B, dtype=model_dtype, device_map="auto").eval()
    model_8b = AutoModelForCausalLM.from_pretrained(MODEL_8B, dtype=model_dtype, device_map="auto").eval()
    model_device = next(model_8b.parameters()).device
    adapter, md = load_adapter(args.checkpoint, model_device, adapter_dtype)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in prompts:
            try:
                res = run_one(item["prompt"], tokenizer, model_1b, model_8b, adapter, model_device, adapter_dtype, model_dtype)
                rec = {**item, "method": args.method_name, "projection_checkpoint": args.checkpoint,
                       "projection_metadata": md, **res, "transfer_success": True, "error": None}
            except Exception as e:
                rec = {**item, "method": args.method_name, "projection_checkpoint": args.checkpoint,
                       "projection_metadata": None, "generated_text": None, "generated_token_ids": None,
                       "prompt_token_count": None, "generation_token_count": None, "step_logits_first5": None,
                       "resume_ttft_ms": None, "ttft_total_ms": None, "prefill_1b_ms": None,
                       "transform_ms": None, "decode_8b_ms": None, "total_latency_ms": None,
                       "transfer_success": False, "error": str(e)}
            f.write(json.dumps(rec) + "\n")
            print(f"{args.method_name}: finished prompt {item['prompt_id']}", flush=True)
    print(f"Saved results to {args.output_file}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/kv_target_layer_adapter.pt")
    p.add_argument("--output_file", default="outputs/transfer_target_layer_aware_mse.jsonl")
    p.add_argument("--prompt_file", default=PROMPT_FILE)
    p.add_argument("--method_name", default="target_layer_aware_mse")
    p.add_argument("--max_prompts", type=int, default=None)
    main(p.parse_args())
