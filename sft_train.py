import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from train_gpt import GPT

PROMPT_TEMPLATE = "### Instruction:\n{inst}\n\n### Response:\n"

class SFTDataset(Dataset):
    def __init__(self, path, sp, block_size):
        self.sp = sp
        self.block_size = block_size
        self.items = []

        bos = sp.bos_id()
        eos = sp.eos_id()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                inst = obj["instruction"].strip()
                resp = obj["response"].strip()

                prompt = PROMPT_TEMPLATE.format(inst=inst)
                full = prompt + resp

                prompt_ids = sp.encode(prompt, out_type=int)
                full_ids = sp.encode(full, out_type=int)

                if bos != -1:
                    prompt_ids = [bos] + prompt_ids
                    full_ids = [bos] + full_ids
                if eos != -1:
                    full_ids = full_ids + [eos]

                # truncate to block_size+1 because we will shift (x=[:-1], y=[1:])
                full_ids = full_ids[: block_size + 1]
                if len(full_ids) < 2:
                    continue

                # SHIFTED LM: x predicts y (next token)
                x = full_ids[:-1]
                y = full_ids[1:]

                # mask prompt part: we only train on response tokens
                # response begins right after prompt end -> first response token is predicted at position (prompt_len-1)
                prompt_len = min(len(prompt_ids), len(full_ids))
                # y index corresponds to predicting token at position t+1; so ignore y positions < (prompt_len-1)
                ignore_upto = max(0, prompt_len - 1)
                y = [-100] * ignore_upto + y[ignore_upto:]

                # ensure same length
                y = y[: len(x)]

                self.items.append((torch.tensor(x, dtype=torch.long),
                                   torch.tensor(y, dtype=torch.long)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_out = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    y_out = torch.full((len(xs), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_out[i, : x.size(0)] = x
        y_out[i, : y.size(0)] = y
    return x_out, y_out


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, targets=None)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", type=str, default="checkpoints_v3/best.pt")
    ap.add_argument("--spm_model", type=str, default="tokenizer/spm.model")
    ap.add_argument("--train_jsonl", type=str, default="data/sft/train.jsonl")
    ap.add_argument("--out_dir", type=str, default="checkpoints_sft")
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)
    pad_id = sp.pad_id()
    if pad_id == -1:
        pad_id = 0

    ckpt = torch.load(args.base_ckpt, map_location=device)
    cfg = ckpt["config"]
    meta = ckpt["meta"]
    vocab_size = int(meta["vocab_size"])

    # keep model architecture from pretrained
    model = GPT(
        vocab_size=vocab_size,
        block_size=min(int(cfg["block_size"]), int(args.block_size)),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ds = SFTDataset(args.train_jsonl, sp, model.block_size)

    n = len(ds)
    n_val = max(1, int(0.05 * n))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n - n_val, n_val],
                                                     generator=torch.Generator().manual_seed(1337))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, pad_id))

    best = float("inf")
    step = 0

    for ep in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, targets=None)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            if (step + 1) % 100 == 0:
                print(f"step {step+1} | loss {loss.item():.4f}")
            step += 1

        v = eval_loss(model, val_loader, device)
        print(f"[epoch {ep+1}] val_loss={v:.4f}")

        torch.save({"model": model.state_dict(), "config": cfg, "meta": meta, "sft": vars(args)}, out_dir / "last.pt")
        if v < best:
            best = v
            torch.save({"model": model.state_dict(), "config": cfg, "meta": meta, "sft": vars(args)}, out_dir / "best.pt")
            print(f"✅ new best saved: {out_dir/'best.pt'} (val_loss={best:.4f})")

    print("SFT done.")

if __name__ == "__main__":
    main()
