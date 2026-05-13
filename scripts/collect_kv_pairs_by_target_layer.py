import os, json, argparse, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_1B = "meta-llama/Llama-3.2-1B"
MODEL_8B = "meta-llama/Llama-3.1-8B"
PROMPT_FILE = "prompts/prompts.jsonl"


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_rows(x, y, max_samples):
    n = x.shape[0]
    if n <= max_samples:
        return x, y
    idx = torch.randperm(n, device=x.device)[:max_samples]
    return x[idx], y[idx]


def main(args):
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(args.seed); random.seed(args.seed)

    print(f"Using device: {device}", flush=True)
    prompts = load_prompts(args.prompt_file)
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    print(f"Using {len(prompts)} prompts", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_8B)
    print("Loading 1B model...", flush=True)
    model_1b = AutoModelForCausalLM.from_pretrained(MODEL_1B, dtype=dtype, device_map="auto").eval()
    print("Loading 8B model...", flush=True)
    model_8b = AutoModelForCausalLM.from_pretrained(MODEL_8B, dtype=dtype, device_map="auto").eval()

    layers = {j: {"key_x": [], "key_y": [], "value_x": [], "value_y": [],
                  "source_layer_idx": j // 2, "target_layer_idx": j} for j in range(32)}

    with torch.no_grad():
        for pidx, item in enumerate(prompts):
            inputs = tokenizer(item["prompt"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out_1b = model_1b(**inputs, use_cache=True)
            out_8b = model_8b(**inputs, use_cache=True)
            pkv_1b, pkv_8b = out_1b.past_key_values, out_8b.past_key_values

            for src_i, (k1, v1) in enumerate(pkv_1b):
                for tgt_i in (2 * src_i, 2 * src_i + 1):
                    k8, v8 = pkv_8b[tgt_i]
                    k1f, k8f = k1.reshape(-1, k1.shape[-1]), k8.reshape(-1, k8.shape[-1])
                    v1f, v8f = v1.reshape(-1, v1.shape[-1]), v8.reshape(-1, v8.shape[-1])
                    k1s, k8s = sample_rows(k1f, k8f, args.samples_per_layer)
                    v1s, v8s = sample_rows(v1f, v8f, args.samples_per_layer)
                    layers[tgt_i]["key_x"].append(k1s.cpu())
                    layers[tgt_i]["key_y"].append(k8s.cpu())
                    layers[tgt_i]["value_x"].append(v1s.cpu())
                    layers[tgt_i]["value_y"].append(v8s.cpu())

            print(f"Collected target-layer-aware KV pairs for prompt {pidx + 1}/{len(prompts)}", flush=True)

    packed = {}
    for tgt_i, d in layers.items():
        packed[tgt_i] = {
            "key_x": torch.cat(d["key_x"], dim=0),
            "key_y": torch.cat(d["key_y"], dim=0),
            "value_x": torch.cat(d["value_x"], dim=0),
            "value_y": torch.cat(d["value_y"], dim=0),
            "source_layer_idx": d["source_layer_idx"],
            "target_layer_idx": d["target_layer_idx"],
        }
        print(f"Layer {tgt_i:02d}: key_x {packed[tgt_i]['key_x'].shape}, key_y {packed[tgt_i]['key_y'].shape}", flush=True)

    dataset = {"layers": packed, "metadata": {
        "model_1b": MODEL_1B, "model_8b": MODEL_8B, "num_prompts": len(prompts),
        "samples_per_layer": args.samples_per_layer, "source_dim": 64, "target_dim": 128,
        "num_source_layers": 16, "num_target_layers": 32,
        "layer_mapping": "1B layer i -> 8B layers 2i and 2i+1, stored separately by target layer",
        "prompt_file": args.prompt_file}}
    torch.save(dataset, args.output_file)
    print(f"Saved target-layer-aware KV pair dataset to {args.output_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", default=PROMPT_FILE)
    parser.add_argument("--output_file", default="data/kv_pairs_by_target_layer.pt")
    parser.add_argument("--max_prompts", type=int, default=80)
    parser.add_argument("--samples_per_layer", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
