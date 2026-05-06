import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split


class KVProjection(nn.Module):
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


def train_one_mapper(name, x, y, args, device):
    x = x.float()
    y = y.float()

    dataset = TensorDataset(x, y)

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    mapper = KVProjection(
        input_dim=64,
        output_dim=128,
        hidden_dim=args.hidden_dim
    ).to(device)

    optimizer = torch.optim.AdamW(
        mapper.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        mapper.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = mapper(batch_x)
            loss = F.mse_loss(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_x.shape[0]

        avg_train_loss = total_train_loss / train_size

        mapper.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred = mapper(batch_x)
                loss = F.mse_loss(pred, batch_y)

                total_val_loss += loss.item() * batch_x.shape[0]

        avg_val_loss = total_val_loss / val_size

        print(
            f"{name} mapper | epoch {epoch + 1}/{args.epochs} | "
            f"train MSE: {avg_train_loss:.6f} | val MSE: {avg_val_loss:.6f}",
            flush=True
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in mapper.state_dict().items()
            }

    mapper.load_state_dict(best_state)

    return mapper, best_val_loss


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print(f"Loading dataset from {args.data_file}", flush=True)
    data = torch.load(args.data_file, map_location="cpu")

    key_x = data["key_x"]
    key_y = data["key_y"]
    value_x = data["value_x"]
    value_y = data["value_y"]

    print(f"key_x: {key_x.shape}", flush=True)
    print(f"key_y: {key_y.shape}", flush=True)
    print(f"value_x: {value_x.shape}", flush=True)
    print(f"value_y: {value_y.shape}", flush=True)

    print("Training key mapper...", flush=True)
    key_mapper, key_val_loss = train_one_mapper(
        "key", key_x, key_y, args, device
    )

    print("Training value mapper...", flush=True)
    value_mapper, value_val_loss = train_one_mapper(
        "value", value_x, value_y, args, device
    )

    checkpoint = {
        "key_mapper_state_dict": key_mapper.state_dict(),
        "value_mapper_state_dict": value_mapper.state_dict(),
        "key_val_loss": key_val_loss,
        "value_val_loss": value_val_loss,
        "metadata": {
            "input_dim": 64,
            "output_dim": 128,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "source_data": args.data_file,
            "dataset_metadata": data.get("metadata", {})
        }
    }

    output_path = os.path.join(args.output_dir, "kv_projection.pt")
    torch.save(checkpoint, output_path)

    print(f"Saved learned projection checkpoint to {output_path}", flush=True)
    print(f"Best key val MSE: {key_val_loss:.6f}", flush=True)
    print(f"Best value val MSE: {value_val_loss:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data/kv_pairs.pt")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
