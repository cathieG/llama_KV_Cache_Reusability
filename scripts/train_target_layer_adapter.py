import os, argparse, torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from adapter_common import TargetLayerAwareKVAdapter


def train_one(name, mapper, x, y, args, device):
    x, y = x.float(), y.float()
    ds = TensorDataset(x, y)
    val_size = max(1, int(len(ds) * args.val_ratio))
    train_size = len(ds) - val_size
    train_set, val_set = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        mapper.train(); total = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            loss = F.mse_loss(mapper(bx), by)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item() * bx.shape[0]
        avg_train = total / train_size
        mapper.eval(); total = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                total += F.mse_loss(mapper(bx), by).item() * bx.shape[0]
        avg_val = total / val_size
        print(f"{name} | epoch {epoch+1}/{args.epochs} | train MSE: {avg_train:.6f} | val MSE: {avg_val:.6f}", flush=True)
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in mapper.state_dict().items()}
    mapper.load_state_dict(best_state)
    return best_val


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"Using device: {device}", flush=True)
    data = torch.load(args.data_file, map_location="cpu")
    layers = data["layers"]
    adapter = TargetLayerAwareKVAdapter(hidden_dim=args.hidden_dim).to(device)
    key_losses, value_losses = {}, {}
    for tgt_i in range(32):
        d = layers[tgt_i]
        print(f"\n=== Target layer {tgt_i:02d} key ===", flush=True)
        key_losses[str(tgt_i)] = train_one(f"layer_{tgt_i:02d}_key", adapter.key_mappers[tgt_i], d["key_x"], d["key_y"], args, device)
        print(f"\n=== Target layer {tgt_i:02d} value ===", flush=True)
        value_losses[str(tgt_i)] = train_one(f"layer_{tgt_i:02d}_value", adapter.value_mappers[tgt_i], d["value_x"], d["value_y"], args, device)
    ckpt = {
        "adapter_state_dict": adapter.state_dict(),
        "key_val_losses": key_losses,
        "value_val_losses": value_losses,
        "metadata": {
            "adapter_type": "target_layer_aware",
            "training_objective": "MSE reconstruction of native 8B KV by target layer",
            "num_target_layers": 32, "num_source_layers": 16,
            "input_dim": 64, "output_dim": 128, "hidden_dim": args.hidden_dim,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "source_data": args.data_file,
            "dataset_metadata": data.get("metadata", {}),
            "layer_mapping": "1B layer i -> separately learned mappers for 8B layers 2i and 2i+1"
        }
    }
    out = os.path.join(args.output_dir, args.output_name)
    torch.save(ckpt, out)
    print(f"Saved target-layer-aware adapter checkpoint to {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", default="data/kv_pairs_by_target_layer.pt")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--output_name", default="kv_target_layer_adapter.pt")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
