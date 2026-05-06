import os
import json
import argparse
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_1B = "meta-llama/Llama-3.2-1B"
MODEL_8B = "meta-llama/Llama-3.1-8B"
PROMPT_FILE = "prompts/prompts.jsonl"


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def sample_rows(x, y, max_samples):
    """
    x shape: [N, 64]
    y shape: [N, 128]
    """
    n = x.shape[0]

    if n <= max_samples:
        return x, y

    idx = torch.randperm(n, device=x.device)[:max_samples]
    return x[idx], y[idx]


def main(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("Loading prompts...", flush=True)
    prompts = load_prompts(PROMPT_FILE)

    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]

    print(f"Using {len(prompts)} prompts", flush=True)

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

    all_key_x = []
    all_key_y = []
    all_value_x = []
    all_value_y = []

    with torch.no_grad():
        for prompt_idx, item in enumerate(prompts):
            prompt = item["prompt"]

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out_1b = model_1b(**inputs, use_cache=True)
            out_8b = model_8b(**inputs, use_cache=True)

            pkv_1b = out_1b.past_key_values
            pkv_8b = out_8b.past_key_values

            # 1B has 16 layers; 8B has 32 layers.
            # Pair each 1B layer with two corresponding 8B layers.
            for layer_idx in range(len(pkv_1b)):
                k1, v1 = pkv_1b[layer_idx]

                for target_layer_idx in [2 * layer_idx, 2 * layer_idx + 1]:
                    k8, v8 = pkv_8b[target_layer_idx]

                    # Flatten batch, heads, seq_len into rows.
                    # k1: [1, 8, seq_len, 64]  -> [N, 64]
                    # k8: [1, 8, seq_len, 128] -> [N, 128]
                    k1_flat = k1.reshape(-1, k1.shape[-1])
                    k8_flat = k8.reshape(-1, k8.shape[-1])

                    v1_flat = v1.reshape(-1, v1.shape[-1])
                    v8_flat = v8.reshape(-1, v8.shape[-1])

                    k1_sample, k8_sample = sample_rows(
                        k1_flat, k8_flat, args.samples_per_layer
                    )
                    v1_sample, v8_sample = sample_rows(
                        v1_flat, v8_flat, args.samples_per_layer
                    )

                    all_key_x.append(k1_sample.detach().cpu())
                    all_key_y.append(k8_sample.detach().cpu())
                    all_value_x.append(v1_sample.detach().cpu())
                    all_value_y.append(v8_sample.detach().cpu())

            print(
                f"Collected KV pairs for prompt {prompt_idx + 1}/{len(prompts)}",
                flush=True
            )

    dataset = {
        "key_x": torch.cat(all_key_x, dim=0),
        "key_y": torch.cat(all_key_y, dim=0),
        "value_x": torch.cat(all_value_x, dim=0),
        "value_y": torch.cat(all_value_y, dim=0),
        "metadata": {
            "model_1b": MODEL_1B,
            "model_8b": MODEL_8B,
            "num_prompts": len(prompts),
            "samples_per_layer": args.samples_per_layer,
            "source_dim": 64,
            "target_dim": 128,
            "layer_mapping": "1B layer i -> 8B layers 2i and 2i+1"
        }
    }

    torch.save(dataset, args.output_file)

    print(f"Saved KV pair dataset to {args.output_file}", flush=True)
    print(f"key_x shape: {dataset['key_x'].shape}", flush=True)
    print(f"key_y shape: {dataset['key_y'].shape}", flush=True)
    print(f"value_x shape: {dataset['value_x'].shape}", flush=True)
    print(f"value_y shape: {dataset['value_y'].shape}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="data/kv_pairs.pt")
    parser.add_argument("--max_prompts", type=int, default=80)
    parser.add_argument("--samples_per_layer", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
