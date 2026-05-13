import os, json, argparse, random, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from adapter_common import TargetLayerAwareKVAdapter

MODEL_1B = "meta-llama/Llama-3.2-1B"
MODEL_8B = "meta-llama/Llama-3.1-8B"
PROMPT_FILE = "prompts/prompts.jsonl"


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_prompts(prompts, val_ratio, seed):
    rng = random.Random(seed); prompts = list(prompts); rng.shuffle(prompts)
    val_size = int(len(prompts) * val_ratio)
    return prompts[val_size:], prompts[:val_size]


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def load_adapter(path, device):
    ckpt = torch.load(path, map_location="cpu")
    md = ckpt["metadata"]
    adapter = TargetLayerAwareKVAdapter(
        num_target_layers=md.get("num_target_layers", 32),
        input_dim=md.get("input_dim", 64), output_dim=md.get("output_dim", 128),
        hidden_dim=md.get("hidden_dim", 256))
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    return adapter.to(device=device, dtype=torch.float32).train(), md


def transform_trainable(pkv_1b, adapter, device, model_dtype):
    transformed = []
    for src_i, (k, v) in enumerate(pkv_1b):
        k = k.detach().to(device=device, dtype=torch.float32)
        v = v.detach().to(device=device, dtype=torch.float32)
        for tgt_i in (2 * src_i, 2 * src_i + 1):
            k2, v2 = adapter.forward_target_layer(tgt_i, k, v)
            transformed.append((k2.to(dtype=model_dtype), v2.to(dtype=model_dtype)))
    return DynamicCache.from_legacy_cache(tuple(transformed))


def attn_mask_for_step(past, cur, dtype, device):
    return torch.ones((cur.shape[0], past.get_seq_length() + cur.shape[1]), dtype=dtype, device=device)


def kl_from_logits(teacher_logits, student_logits, temp):
    p = F.softmax(teacher_logits.float() / temp, dim=-1)
    qlog = F.log_softmax(student_logits.float() / temp, dim=-1)
    return F.kl_div(qlog, p, reduction="batchmean") * (temp ** 2)


def compute_loss(prompt, tokenizer, model_1b, model_8b, adapter, args, device, model_dtype):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    if input_ids.shape[1] < 1:
        raise ValueError("Prompt tokenization produced empty input.")

    with torch.no_grad():
        out_1b = model_1b(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        teacher = model_8b(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        teacher_logits = teacher.logits[:, -1, :]
        teacher_past = teacher.past_key_values

    student_past = transform_trainable(out_1b.past_key_values, adapter, device, model_dtype)
    cur = input_ids[:, -1:]  # match existing evaluation resume convention
    losses = []

    for step in range(args.kl_steps):
        smask = attn_mask_for_step(student_past, cur, attention_mask.dtype, device)
        student = model_8b(input_ids=cur, attention_mask=smask, past_key_values=student_past, use_cache=True)
        losses.append(kl_from_logits(teacher_logits, student.logits[:, -1, :], args.temperature))

        with torch.no_grad():
            next_teacher = torch.argmax(teacher_logits, dim=-1, keepdim=True)
        cur = next_teacher
        student_past = student.past_key_values

        if step < args.kl_steps - 1:
            with torch.no_grad():
                tmask = attn_mask_for_step(teacher_past, cur, attention_mask.dtype, device)
                teacher = model_8b(input_ids=cur, attention_mask=tmask, past_key_values=teacher_past, use_cache=True)
                teacher_logits = teacher.logits[:, -1, :]
                teacher_past = teacher.past_key_values

    return torch.stack(losses).mean()


def evaluate(prompts, tokenizer, model_1b, model_8b, adapter, args, device, model_dtype):
    if not prompts:
        return None
    adapter.eval(); vals = []
    with torch.no_grad():
        for item in prompts:
            vals.append(float(compute_loss(item["prompt"], tokenizer, model_1b, model_8b, adapter, args, device, model_dtype).item()))
    adapter.train()
    return sum(vals) / len(vals)


def save_ckpt(path, adapter, args, init_md, best_val, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "adapter_state_dict": adapter.state_dict(),
        "kl_val_loss": best_val,
        "metadata": {
            "adapter_type": "target_layer_aware",
            "training_objective": "KL divergence between native 8B logits and transferred-cache 8B logits",
            "num_target_layers": 32, "num_source_layers": 16,
            "input_dim": 64, "output_dim": 128,
            "hidden_dim": init_md.get("hidden_dim", args.hidden_dim),
            "kl_steps": args.kl_steps, "temperature": args.temperature,
            "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip, "seed": args.seed,
            "init_checkpoint": args.init_checkpoint, "init_metadata": init_md,
            "model_1b": MODEL_1B, "model_8b": MODEL_8B,
            "layer_mapping": "1B layer i -> separate mappers for target 8B layers 2i and 2i+1"
        },
        "train_history": history
    }, path)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}", flush=True); print(f"Model dtype: {model_dtype}", flush=True)

    prompts = load_prompts(args.prompt_file)
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    train_prompts, val_prompts = split_prompts(prompts, args.val_ratio, args.seed)
    print(f"Train prompts: {len(train_prompts)}", flush=True); print(f"Val prompts: {len(val_prompts)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_8B)
    model_1b = AutoModelForCausalLM.from_pretrained(MODEL_1B, dtype=model_dtype, device_map="auto")
    model_8b = AutoModelForCausalLM.from_pretrained(MODEL_8B, dtype=model_dtype, device_map="auto")
    freeze_model(model_1b); freeze_model(model_8b)
    model_device = next(model_8b.parameters()).device
    print(f"8B first-parameter device: {model_device}", flush=True)

    adapter, init_md = load_adapter(args.init_checkpoint, model_device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_state, history, global_step = float("inf"), None, [], 0

    for epoch in range(args.epochs):
        random.shuffle(train_prompts); running = 0.0
        for item in train_prompts:
            loss = compute_loss(item["prompt"], tokenizer, model_1b, model_8b, adapter, args, model_device, model_dtype)
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            opt.step()
            running += float(loss.item()); global_step += 1
            if global_step % args.log_every == 0:
                print(f"epoch {epoch+1}/{args.epochs} | step {global_step} | train KL: {running/global_step:.6f}", flush=True)
            if device == "cuda" and global_step % args.empty_cache_every == 0:
                torch.cuda.empty_cache()
        avg_train = running / max(len(train_prompts), 1)
        val = evaluate(val_prompts, tokenizer, model_1b, model_8b, adapter, args, model_device, model_dtype)
        print(f"epoch {epoch+1}/{args.epochs} complete | avg train KL: {avg_train:.6f} | val KL: {val if val is not None else 'N/A'}", flush=True)
        history.append({"epoch": epoch+1, "avg_train_kl": avg_train, "val_kl": val})
        score = avg_train if val is None else val
        if score < best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            adapter.load_state_dict(best_state)
            out = os.path.join(args.output_dir, args.output_name)
            save_ckpt(out, adapter, args, init_md, best_val, history)
            adapter.to(device=model_device, dtype=torch.float32).train()
            print(f"Saved best target-layer-aware KL adapter to {out}", flush=True)
    if best_state is not None:
        adapter.load_state_dict(best_state)
    out = os.path.join(args.output_dir, args.output_name)
    save_ckpt(out, adapter, args, init_md, best_val, history)
    print(f"Training complete. Final checkpoint saved to {out}", flush=True)
    print(f"Best validation/selection KL: {best_val:.6f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt_file", default=PROMPT_FILE)
    p.add_argument("--init_checkpoint", default="checkpoints/kv_target_layer_adapter.pt")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--output_name", default="kv_target_layer_adapter_kl.pt")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--kl_steps", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_prompts", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--empty_cache_every", type=int, default=10)
    main(p.parse_args())
