import io
import json
import argparse
from pathlib import Path
from array import array

import zstandard as zstd
import sentencepiece as spm
from tqdm import tqdm


def iter_jsonl_zst(path: Path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default="data/clean/train.jsonl.zst")
    ap.add_argument("--spm_model", type=str, default="tokenizer/spm.model")
    ap.add_argument("--out_dir", type=str, default="data/tokens")
    ap.add_argument("--val_ratio", type=float, default=0.01)
    ap.add_argument("--max_tokens_per_doc", type=int, default=2048)
    ap.add_argument("--add_bos_eos", action="store_true")
    args = ap.parse_args()

    infile = Path(args.infile)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(Path(args.spm_model)))

    vocab_size = sp.get_piece_size()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    # We will store uint16 because vocab is 16k (fits in 0..65535)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    # streaming write
    train_f = open(train_path, "wb")
    val_f = open(val_path, "wb")

    n_docs = 0
    n_train_tok = 0
    n_val_tok = 0

    # pseudo-random split without loading everything: hash(url) based
    # (stable split, avoids leakage across reruns)
    def goes_to_val(url: str) -> bool:
        # deterministic: use last bytes of hash
        return (hash(url) & 0xFFFFFFFF) / 0xFFFFFFFF < args.val_ratio

    it = iter_jsonl_zst(infile)

    for obj in tqdm(it, desc="Encoding docs"):
        text = obj.get("text", "")
        url = obj.get("url", "")

        ids = sp.encode(text, out_type=int)
        if args.max_tokens_per_doc and len(ids) > args.max_tokens_per_doc:
            ids = ids[: args.max_tokens_per_doc]

        if args.add_bos_eos:
            if bos_id != -1:
                ids = [bos_id] + ids
            if eos_id != -1:
                ids = ids + [eos_id]

        if not ids:
            continue

        # write as uint16
        arr = array("H", ids)  # unsigned short = uint16
        b = arr.tobytes()

        if goes_to_val(url):
            val_f.write(b)
            n_val_tok += len(ids)
        else:
            train_f.write(b)
            n_train_tok += len(ids)

        n_docs += 1

    train_f.close()
    val_f.close()

    meta = {
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "val_ratio": args.val_ratio,
        "max_tokens_per_doc": args.max_tokens_per_doc,
        "add_bos_eos": bool(args.add_bos_eos),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Docs: {n_docs:,}")
    print(f"Train tokens: {n_train_tok:,}")
    print(f"Val tokens:   {n_val_tok:,}")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
