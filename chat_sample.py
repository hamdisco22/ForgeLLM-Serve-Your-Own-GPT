import argparse
import torch
import sentencepiece as spm
from train_gpt import GPT

PROMPT_TEMPLATE = "### Instruction:\n{inst}\n\n### Response:\n"

@torch.no_grad()
def generate(model, x, max_new_tokens, eos_id=None, temperature=0.8, top_k=50, repetition_penalty=1.1):
    model.eval()
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :]

        # repetition penalty
        if repetition_penalty and repetition_penalty > 1.0:
            for b in range(x.size(0)):
                used = set(x[b].tolist())
                logits[b, list(used)] /= repetition_penalty

        logits = logits / max(1e-6, temperature)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_sft/best.pt")
    ap.add_argument("--spm_model", default="tokenizer/spm.model")
    ap.add_argument("--instruction", default="Explain neural networks in 5 lines.")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    meta = ckpt["meta"]

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)
    eos_id = sp.eos_id() if sp.eos_id() != -1 else None
    bos_id = sp.bos_id()

    prompt = PROMPT_TEMPLATE.format(inst=args.instruction)
    ids = sp.encode(prompt, out_type=int)
    if bos_id != -1:
        ids = [bos_id] + ids

    x = torch.tensor([ids], dtype=torch.long, device=device)

    model = GPT(
        vocab_size=int(meta["vocab_size"]),
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    y = generate(model, x, args.max_new_tokens, eos_id=eos_id, temperature=0.25, top_k=30, repetition_penalty=1.2)
    out_ids = y[0].tolist()

# decode only newly generated tokens (after prompt)
    new_ids = out_ids[len(ids):]

# cut at EOS if present
    if eos_id is not None and eos_id in new_ids:
        new_ids = new_ids[:new_ids.index(eos_id)]

    answer = sp.decode(new_ids).strip()

# remove accidental section headers if they appear
    if "###" in answer:
        answer = answer.split("###", 1)[0].strip()

    answer = answer.lstrip(" \n\r\t:=-,")
    print(answer.strip())


if __name__ == "__main__":
    main()
