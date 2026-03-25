import json, argparse
from pathlib import Path
import numpy as np
import torch

from train_gpt import GPT, load_bin, get_batch

@torch.no_grad()
def estimate(model, data, block_size, batch_size, device, iters=200):
    model.eval()
    losses = torch.zeros(iters, device=device)
    for k in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss
    return losses.mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--test_bin", type=str, required=True)
    ap.add_argument("--meta", type=str, required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    vocab_size = int(meta["vocab_size"])

    model = GPT(
        vocab_size=vocab_size,
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
    test_data = load_bin(Path(args.test_bin), dtype=dtype)

    test_loss = estimate(model, test_data, int(cfg["block_size"]), args.batch_size, device, args.iters)
    print("TEST loss:", test_loss)

if __name__ == "__main__":
    main()
