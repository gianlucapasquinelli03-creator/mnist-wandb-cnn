"""
Training script per la CNN su MNIST, con tracking completo su Weights & Biases.

Esempio d'uso:
    python train/train.py --epochs 10 --batch_size 64 --lr 1e-3

La prima volta ti chiederà di fare login su W&B (wandb login),
o di inserire la tua API key (https://wandb.ai/authorize).
"""

import argparse
import os
import subprocess
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.mnist import get_dataloaders
from models.cnn import SimpleCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST with W&B tracking")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--project", type=str, default="mnist-cnn")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--tags", type=str, nargs="*", default=[])
    parser.add_argument("--log_freq", type=int, default=50, help="ogni quanti step loggare la batch loss")
    parser.add_argument("--no_wandb", action="store_true", help="disabilita W&B (utile per debug rapido)")
    return parser.parse_args()


def get_git_commit() -> str:
    """Recupera l'hash del commit corrente, per collegare la run al codice esatto usato."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "disabled" if args.no_wandb else "online"
    wandb.init(
        project=args.project,
        name=args.run_name,
        tags=args.tags,
        mode=mode,
        config={
            **vars(args),
            "git_commit": get_git_commit(),
            "device": str(device),
        },
    )
    config = wandb.config

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        val_split=config.val_split,
        seed=config.seed,
    )

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    # Traccia gradienti e pesi durante il training (visibile nella dashboard W&B)
    wandb.watch(model, log="all", log_freq=100)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
            global_step += 1

            if global_step % config.log_freq == 0:
                wandb.log({"train/batch_loss": loss.item()}, step=global_step)
                pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        }, step=global_step)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Salva il checkpoint migliore e lo versiona come Artifact W&B
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)

            artifact = wandb.Artifact(
                name="mnist-cnn-model",
                type="model",
                metadata={"val_accuracy": val_acc, "epoch": epoch + 1},
            )
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact, aliases=["best", "latest"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    wandb.log({"test/loss": test_loss, "test/accuracy": test_acc}, step=global_step)
    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.summary["test_accuracy"] = test_acc

    print(f"Test finale: loss={test_loss:.4f} acc={test_acc:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
