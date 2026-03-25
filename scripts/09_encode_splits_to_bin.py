import io, json, argparse
from pathlib import Path
from array import array

import zstandard as zstd
import sentencepiece as spm
from tqdm import tqdm

def iter_jsonl_zst(path: Path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            ts = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in ts:
                line = line.strip()
                if line:
                    yield json.loads(line)

def encode_file(infile: Path, sp, outfile: Path, max_tokens_per_doc=2048, add_bos_eos=True):
    bos_id, eos_id = sp.bos_id(), sp.eos_id()
    total = 0
    with open(outfile, "wb") as f:
        for obj in tqdm(iter_jsonl_zst(infile), desc=f"Encoding {infile.name}"):
            text = obj.get("text", "")
            ids = sp.encode(text, out_type=int)
            if max_tokens_per_doc and len(ids) > max_tokens_per_doc:
                ids = ids[:max_tokens_per_doc]
            if add_bos_eos:
                if bos_id != -1: ids = [bos_id] + ids
                if eos_id != -1: ids = ids + [eos_id]
            if not ids:
                continue
            arr = array("H", ids)  # uint16
            f.write(arr.tobytes())
            total += len(ids)
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", type=str, required=True)
    ap.add_argument("--val_in", type=str, required=True)
    ap.add_argument("--spm_model", type=str, default="tokenizer/spm.model")
    ap.add_argument("--out_dir", type=str, default="data/tokens_v2")
    ap.add_argument("--max_tokens_per_doc", type=int, default=2048)
    ap.add_argument("--add_bos_eos", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)

    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"

    train_tok = encode_file(Path(args.train_in), sp, train_bin, args.max_tokens_per_doc, args.add_bos_eos)
    val_tok = encode_file(Path(args.val_in), sp, val_bin, args.max_tokens_per_doc, args.add_bos_eos)

    meta = {
        "vocab_size": sp.get_piece_size(),
        "dtype": "uint16",
        "max_tokens_per_doc": args.max_tokens_per_doc,
        "add_bos_eos": bool(args.add_bos_eos),
        "train_tokens": train_tok,
        "val_tokens": val_tok,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nDone.")
    print("Train tokens:", train_tok)
    print("Val tokens  :", val_tok)
    print("Saved:", out_dir)

if __name__ == "__main__":
    main()
