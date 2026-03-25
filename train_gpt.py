import os
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Model: GPT (decoder-only)
# -------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer("mask", None, persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        if self.mask is None or self.mask.size(-1) < T:
            # causal mask (T x T)
            m = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            self.mask = m

        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# -------------------------
# Data: memmap uint16 bins
# -------------------------

def load_bin(path: Path, dtype=np.uint16):
    return np.memmap(str(path), dtype=dtype, mode="r")

def get_batch(data, block_size, batch_size, device):
    # data is 1D array of token ids
    n = len(data)
    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/tokens")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    vocab_size = int(meta["vocab_size"])
    dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32

    train_data = load_bin(data_dir / "train.bin", dtype=dtype)
    val_data = load_bin(data_dir / "val.bin", dtype=dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"

    model = GPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step = 0
    best_val = float("inf")

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from step {start_step}")

    t0 = time.time()
    model.train()

    for step in range(start_step, args.max_steps):
        optimizer.zero_grad(set_to_none=True)

        # gradient accumulation
        total_loss = 0.0
        for micro in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % 50 == 0:
            dt = time.time() - t0
            tok_per_step = args.batch_size * args.grad_accum * args.block_size
            print(f"step {step+1}/{args.max_steps} | loss {total_loss:.4f} | {dt:.1f}s | tokens/step {tok_per_step}")
            t0 = time.time()

        if (step + 1) % args.eval_every == 0 or step == 0:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, device, args.eval_iters)
            print(f"[eval] step {step+1}: train {losses['train']:.4f} | val {losses['val']:.4f}")

            # save last
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "best_val": best_val,
                    "config": vars(args),
                    "meta": meta,
                },
                ckpt_path,
            )

            # save best
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "best_val": best_val,
                        "config": vars(args),
                        "meta": meta,
                    },
                    best_path,
                )
                print(f"✅ new best val: {best_val:.4f} saved to {best_path}")

    print("Training finished.")
    print("Last checkpoint:", ckpt_path)
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()
