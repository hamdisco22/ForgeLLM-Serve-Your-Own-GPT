import json
import argparse
from pathlib import Path

import torch
import sentencepiece as spm

from train_gpt import GPT  # uses same model class


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-6, temperature)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--spm_model", type=str, default="tokenizer/spm.model")
    ap.add_argument("--prompt", type=str, default="Hello,")
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    meta = ckpt["meta"]
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

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)

    ids = sp.encode(args.prompt, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    y = generate(model, x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)[0].tolist()
    print(sp.decode(y))


if __name__ == "__main__":
    main()
