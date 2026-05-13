import os
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


MODEL_1B = "meta-llama/Llama-3.2-1B"
MODEL_8B = "meta-llama/Llama-3.1-8B"
PROMPT_FILE = "prompts/prompts.jsonl"


class KVProjection(nn.Module):
    """
    Same projection module used in train_kv_projection.py and run_transfer_learned.py.

    This checkpoint format is intentionally kept compatible with run_transfer_learned.py:
      - key_mapper_state_dict
      - value_mapper_state_dict
      - metadata
    """
    def __init__(self, input_dim=64, output_dim=128, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.net(x)


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def split_prompts(prompts, val_ratio, seed):
    rng = random.Random(seed)
    prompts = list(prompts)
    rng.shuffle(prompts)

    val_size = int(len(prompts) * val_ratio)
    val_prompts = prompts[:val_size]
    train_prompts = prompts[val_size:]

    return train_prompts, val_prompts


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def load_or_init_mappers(args, device):
    """
    Loads the current MSE-trained checkpoint if available.
    Otherwise initializes from scratch only when --allow_random_init is passed.
    """
    hidden_dim = args.hidden_dim

    key_mapper = KVProjection(
        input_dim=64,
        output_dim=128,
        hidden_dim=hidden_dim
    ).to(device=device, dtype=torch.float32)

    value_mapper = KVProjection(
        input_dim=64,
        output_dim=128,
        hidden_dim=hidden_dim
    ).to(device=device, dtype=torch.float32)

    init_metadata = {}

    if args.init_checkpoint is not None and os.path.exists(args.init_checkpoint):
        print(f"Loading initial projection from {args.init_checkpoint}", flush=True)
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        metadata = ckpt.get("metadata", {})
        ckpt_hidden_dim = metadata.get("hidden_dim", None)

        if ckpt_hidden_dim != hidden_dim:
            raise ValueError(
                f"Checkpoint hidden_dim={ckpt_hidden_dim}, but args.hidden_dim={hidden_dim}. "
                "Use the same hidden_dim as the original projection checkpoint."
            )

        key_mapper.load_state_dict(ckpt["key_mapper_state_dict"])
        value_mapper.load_state_dict(ckpt["value_mapper_state_dict"])
        init_metadata = metadata

    elif not args.allow_random_init:
        raise FileNotFoundError(
            f"Initial checkpoint not found: {args.init_checkpoint}. "
            "For KL training, it is strongly recommended to initialize from the MSE-trained "
            "projection checkpoint. If you really want random initialization, pass "
            "--allow_random_init."
        )
    else:
        print("WARNING: randomly initializing projection mappers.", flush=True)

    key_mapper.train()
    value_mapper.train()

    return key_mapper, value_mapper, init_metadata


def transform_1b_to_8b_pkv_learned_trainable(
    past_key_values_1b,
    key_mapper,
    value_mapper,
    device,
    model_dtype
):
    """
    Trainable version of the transform used in run_transfer_learned.py.

    Important:
    - 1B KV tensors are detached from the 1B model.
    - Projection runs in float32 for more stable optimization.
    - Projected KV tensors are cast back to the 8B model dtype before being
      passed into the frozen 8B model.
    - We do NOT use torch.no_grad() here, because gradients must flow from
      the 8B logits back into key_mapper/value_mapper.
    """
    transformed = []

    for layer_idx in range(len(past_key_values_1b)):
        k, v = past_key_values_1b[layer_idx]

        # k/v shape: [batch, heads, seq_len, 64]
        k = k.detach().to(device=device, dtype=torch.float32)
        v = v.detach().to(device=device, dtype=torch.float32)

        k2 = key_mapper(k).to(dtype=model_dtype)
        v2 = value_mapper(v).to(dtype=model_dtype)

        # 1B has 16 layers; 8B has 32 layers.
        # Keep your original mapping: 1B layer i -> 8B layers 2i and 2i+1.
        transformed.append((k2, v2))
        transformed.append((k2.clone(), v2.clone()))

    return DynamicCache.from_legacy_cache(tuple(transformed))


def attention_mask_for_step(past, current_input_ids, dtype, device):
    past_len = past.get_seq_length()
    return torch.ones(
        (current_input_ids.shape[0], past_len + current_input_ids.shape[1]),
        dtype=dtype,
        device=device
    )


def kl_from_logits(teacher_logits, student_logits, temperature):
    """
    KL(P_teacher || P_student), using temperature-scaled distillation.
    """
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)

    return F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    ) * (temperature ** 2)


def kv_reconstruction_loss(projected_past, native_past):
    """
    Optional stabilizer:
    MSE between projected KV and native 8B KV for the prompt.

    This is not the main objective. Use a very small weight, e.g. 0.001.
    """
    losses = []

    proj_legacy = projected_past.to_legacy_cache()
    native_legacy = native_past.to_legacy_cache() if hasattr(native_past, "to_legacy_cache") else native_past

    for (pk, pv), (nk, nv) in zip(proj_legacy, native_legacy):
        losses.append(F.mse_loss(pk.float(), nk.detach().float()))
        losses.append(F.mse_loss(pv.float(), nv.detach().float()))

    return torch.stack(losses).mean()


def compute_kl_training_loss_for_prompt(
    prompt,
    tokenizer,
    model_1b,
    model_8b,
    key_mapper,
    value_mapper,
    args,
    device,
    model_dtype,
    train=True
):
    """
    Computes average KL over the first args.kl_steps resumed decoding steps.

    Teacher:
      native 8B run.

    Student:
      frozen 8B resumed from projected 1B KV.

    Teacher forcing:
      after step 0, both teacher and student are fed the native 8B greedy token.
      This prevents the student from drifting into a different sequence during training.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    if input_ids.shape[1] < 1:
        raise ValueError("Prompt tokenization produced empty input.")

    # ---------------------------
    # 1B prefill: no gradient
    # ---------------------------
    with torch.no_grad():
        out_1b = model_1b(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
        pkv_1b = out_1b.past_key_values

    # ---------------------------
    # Native 8B prompt forward: teacher step 0 and native prompt KV
    # ---------------------------
    with torch.no_grad():
        teacher_out = model_8b(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
        teacher_logits = teacher_out.logits[:, -1, :]
        teacher_past = teacher_out.past_key_values
        teacher_prompt_past = teacher_out.past_key_values

    # ---------------------------
    # Project 1B KV into 8B-shaped KV
    # ---------------------------
    projected_past = transform_1b_to_8b_pkv_learned_trainable(
        past_key_values_1b=pkv_1b,
        key_mapper=key_mapper,
        value_mapper=value_mapper,
        device=device,
        model_dtype=model_dtype
    )

    # Current project convention:
    # run_transfer_learned.py resumes by feeding the last prompt token while
    # also passing the projected prompt cache. We keep that convention so the
    # training objective matches your existing evaluation script.
    current_token = input_ids[:, -1:]

    student_past = projected_past
    losses = []

    for step in range(args.kl_steps):
        # ---------------------------
        # Student step: gradient flows through projected_past into the mappers
        # ---------------------------
        student_attn_mask = attention_mask_for_step(
            past=student_past,
            current_input_ids=current_token,
            dtype=attention_mask.dtype,
            device=device
        )

        student_out = model_8b(
            input_ids=current_token,
            attention_mask=student_attn_mask,
            past_key_values=student_past,
            use_cache=True
        )

        student_logits = student_out.logits[:, -1, :]

        losses.append(
            kl_from_logits(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                temperature=args.temperature
            )
        )

        # Native 8B greedy token used for teacher forcing.
        with torch.no_grad():
            next_teacher_token = torch.argmax(teacher_logits, dim=-1, keepdim=True)

        # Advance both teacher and student.
        current_token = next_teacher_token
        student_past = student_out.past_key_values

        if step < args.kl_steps - 1:
            with torch.no_grad():
                teacher_attn_mask = attention_mask_for_step(
                    past=teacher_past,
                    current_input_ids=current_token,
                    dtype=attention_mask.dtype,
                    device=device
                )

                teacher_out = model_8b(
                    input_ids=current_token,
                    attention_mask=teacher_attn_mask,
                    past_key_values=teacher_past,
                    use_cache=True
                )

                teacher_logits = teacher_out.logits[:, -1, :]
                teacher_past = teacher_out.past_key_values

    loss_kl = torch.stack(losses).mean()
    loss = loss_kl

    loss_kv = None
    if args.kv_loss_weight > 0:
        # Only compare the prompt-cache portion before either cache is updated by decoding.
        loss_kv = kv_reconstruction_loss(projected_past, teacher_prompt_past)
        loss = loss + args.kv_loss_weight * loss_kv

    return {
        "loss": loss,
        "loss_kl": loss_kl.detach(),
        "loss_kv": None if loss_kv is None else loss_kv.detach()
    }


def evaluate_on_prompts(
    prompts,
    tokenizer,
    model_1b,
    model_8b,
    key_mapper,
    value_mapper,
    args,
    device,
    model_dtype
):
    if not prompts:
        return None

    key_mapper.eval()
    value_mapper.eval()

    losses = []
    kv_losses = []

    # Validation should not build a gradient graph.
    # We temporarily disable grad for the projector too.
    with torch.no_grad():
        for item in prompts:
            prompt = item["prompt"]

            out = compute_kl_training_loss_for_prompt(
                prompt=prompt,
                tokenizer=tokenizer,
                model_1b=model_1b,
                model_8b=model_8b,
                key_mapper=key_mapper,
                value_mapper=value_mapper,
                args=args,
                device=device,
                model_dtype=model_dtype,
                train=False
            )

            losses.append(float(out["loss_kl"].item()))
            if out["loss_kv"] is not None:
                kv_losses.append(float(out["loss_kv"].item()))

    key_mapper.train()
    value_mapper.train()

    return {
        "val_kl": sum(losses) / len(losses),
        "val_kv": None if not kv_losses else sum(kv_losses) / len(kv_losses)
    }


def save_checkpoint(
    output_path,
    key_mapper,
    value_mapper,
    args,
    init_metadata,
    best_val_kl,
    train_history
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    checkpoint = {
        "key_mapper_state_dict": key_mapper.state_dict(),
        "value_mapper_state_dict": value_mapper.state_dict(),
        "key_val_loss": None,
        "value_val_loss": None,
        "kl_val_loss": best_val_kl,
        "metadata": {
            "input_dim": 64,
            "output_dim": 128,
            "hidden_dim": args.hidden_dim,
            "training_objective": "KL divergence between native 8B logits and transferred-cache 8B logits",
            "kl_steps": args.kl_steps,
            "temperature": args.temperature,
            "kv_loss_weight": args.kv_loss_weight,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "init_checkpoint": args.init_checkpoint,
            "init_metadata": init_metadata,
            "model_1b": MODEL_1B,
            "model_8b": MODEL_8B,
            "layer_mapping": "1B layer i -> 8B layers 2i and 2i+1",
            "note": (
                "This checkpoint is compatible with run_transfer_learned.py because it "
                "keeps the same key/value mapper names and architecture."
            )
        },
        "train_history": train_history
    }

    torch.save(checkpoint, output_path)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device}", flush=True)
    print(f"Model dtype: {model_dtype}", flush=True)

    print("Loading prompts...", flush=True)
    prompts = load_prompts(args.prompt_file)

    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]

    train_prompts, val_prompts = split_prompts(
        prompts=prompts,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    print(f"Train prompts: {len(train_prompts)}", flush=True)
    print(f"Val prompts: {len(val_prompts)}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_8B)

    print("Loading 1B model...", flush=True)
    model_1b = AutoModelForCausalLM.from_pretrained(
        MODEL_1B,
        dtype=model_dtype,
        device_map="auto"
    )
    freeze_model(model_1b)

    print("Loading 8B model...", flush=True)
    model_8b = AutoModelForCausalLM.from_pretrained(
        MODEL_8B,
        dtype=model_dtype,
        device_map="auto"
    )
    freeze_model(model_8b)

    # In your current scripts, everything is effectively placed on the model device.
    # This assumes the job is run on one GPU, which matches your existing project setup.
    model_device = next(model_8b.parameters()).device
    print(f"8B first-parameter device: {model_device}", flush=True)

    key_mapper, value_mapper, init_metadata = load_or_init_mappers(
        args=args,
        device=model_device
    )

    optimizer = torch.optim.AdamW(
        list(key_mapper.parameters()) + list(value_mapper.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_kl = float("inf")
    best_key_state = None
    best_value_state = None
    train_history = []

    global_step = 0

    for epoch in range(args.epochs):
        random.shuffle(train_prompts)

        running_kl = 0.0
        running_total = 0

        for idx, item in enumerate(train_prompts):
            prompt = item["prompt"]

            out = compute_kl_training_loss_for_prompt(
                prompt=prompt,
                tokenizer=tokenizer,
                model_1b=model_1b,
                model_8b=model_8b,
                key_mapper=key_mapper,
                value_mapper=value_mapper,
                args=args,
                device=model_device,
                model_dtype=model_dtype,
                train=True
            )

            loss = out["loss"]
            loss_kl = out["loss_kl"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(key_mapper.parameters()) + list(value_mapper.parameters()),
                    args.grad_clip
                )

            optimizer.step()

            running_kl += float(loss_kl.item())
            running_total += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg_kl = running_kl / max(running_total, 1)
                print(
                    f"epoch {epoch + 1}/{args.epochs} | "
                    f"step {global_step} | "
                    f"train KL: {avg_kl:.6f}",
                    flush=True
                )

            # Helps reduce fragmentation when using an 8B model in a loop.
            if device == "cuda" and global_step % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

        avg_train_kl = running_kl / max(running_total, 1)

        val_summary = evaluate_on_prompts(
            prompts=val_prompts,
            tokenizer=tokenizer,
            model_1b=model_1b,
            model_8b=model_8b,
            key_mapper=key_mapper,
            value_mapper=value_mapper,
            args=args,
            device=model_device,
            model_dtype=model_dtype
        )

        val_kl = None if val_summary is None else val_summary["val_kl"]

        print(
            f"epoch {epoch + 1}/{args.epochs} complete | "
            f"avg train KL: {avg_train_kl:.6f} | "
            f"val KL: {val_kl if val_kl is not None else 'N/A'}",
            flush=True
        )

        train_history.append({
            "epoch": epoch + 1,
            "avg_train_kl": avg_train_kl,
            "val_kl": val_kl
        })

        score_for_selection = avg_train_kl if val_kl is None else val_kl

        if score_for_selection < best_val_kl:
            best_val_kl = score_for_selection
            best_key_state = {
                k: v.detach().cpu().clone()
                for k, v in key_mapper.state_dict().items()
            }
            best_value_state = {
                k: v.detach().cpu().clone()
                for k, v in value_mapper.state_dict().items()
            }

            save_path = os.path.join(args.output_dir, args.output_name)
            key_mapper_cpu_state = best_key_state
            value_mapper_cpu_state = best_value_state

            # Save using current best states.
            key_mapper.load_state_dict(best_key_state)
            value_mapper.load_state_dict(best_value_state)
            save_checkpoint(
                output_path=save_path,
                key_mapper=key_mapper,
                value_mapper=value_mapper,
                args=args,
                init_metadata=init_metadata,
                best_val_kl=best_val_kl,
                train_history=train_history
            )

            # Move back to training device.
            key_mapper.to(device=model_device, dtype=torch.float32)
            value_mapper.to(device=model_device, dtype=torch.float32)
            key_mapper.train()
            value_mapper.train()

            print(f"Saved best KL-aligned projection to {save_path}", flush=True)

    if best_key_state is not None:
        key_mapper.load_state_dict(best_key_state)
        value_mapper.load_state_dict(best_value_state)

    final_path = os.path.join(args.output_dir, args.output_name)
    save_checkpoint(
        output_path=final_path,
        key_mapper=key_mapper,
        value_mapper=value_mapper,
        args=args,
        init_metadata=init_metadata,
        best_val_kl=best_val_kl,
        train_history=train_history
    )

    print(f"Training complete. Final checkpoint saved to {final_path}", flush=True)
    print(f"Best validation/selection KL: {best_val_kl:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt_file", default=PROMPT_FILE)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--output_name", default="kv_projection_kl.pt")

    # Recommended: initialize from your existing MSE-trained projection.
    parser.add_argument("--init_checkpoint", default="checkpoints/kv_projection.pt")
    parser.add_argument("--allow_random_init", action="store_true")

    # Must match the original projection architecture.
    parser.add_argument("--hidden_dim", type=int, default=None)

    # KL objective.
    parser.add_argument("--kl_steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kv_loss_weight", type=float, default=0.0)

    # Training.
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Logging / memory.
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--empty_cache_every", type=int, default=10)

    args = parser.parse_args()
    main(args)
