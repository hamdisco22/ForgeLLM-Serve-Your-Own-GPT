import io, json, argparse
from pathlib import Path
from urllib.parse import urlparse

import zstandard as zstd
import xxhash

def iter_jsonl_zst(path: Path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            ts = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in ts:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def domain_of(url: str) -> str:
    try:
        return (urlparse(url).netloc or "unknown").lower()
    except Exception:
        return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="data/clean/splits")
    ap.add_argument("--val_pct", type=int, default=10)
    ap.add_argument("--test_pct", type=int, default=5)
    args = ap.parse_args()

    infile = Path(args.infile)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = list(iter_jsonl_zst(infile))
    if not docs:
        raise SystemExit(f"No docs found in {infile}")

    # collect unique domains
    domains = {}
    for obj in docs:
        d = domain_of(obj.get("url", ""))
        domains[d] = domains.get(d, 0) + 1

    dom_list = list(domains.keys())
    dom_list.sort(key=lambda d: xxhash.xxh3_64_intdigest(d))  # deterministic

    n_dom = len(dom_list)
    k_test = max(1, round(n_dom * args.test_pct / 100)) if n_dom >= 2 else 0
    k_val  = max(1, round(n_dom * args.val_pct  / 100)) if n_dom >= 3 else 0
    if k_test + k_val >= n_dom:
        k_test = 1 if n_dom >= 2 else 0
        k_val = 1 if n_dom >= 3 else 0

    test_set = set(dom_list[:k_test])
    val_set  = set(dom_list[k_test:k_test + k_val])

    train_out = out_dir / "train.jsonl.zst"
    val_out   = out_dir / "val.jsonl.zst"
    test_out  = out_dir / "test.jsonl.zst"

    # IMPORTANT: separate compressors for each output stream
    c_tr = zstd.ZstdCompressor(level=10)
    c_va = zstd.ZstdCompressor(level=10)
    c_te = zstd.ZstdCompressor(level=10)

    counts = {"train": 0, "val": 0, "test": 0}

    with open(train_out, "wb") as ftr, open(val_out, "wb") as fva, open(test_out, "wb") as fte:
        with c_tr.stream_writer(ftr) as wtr, c_va.stream_writer(fva) as wva, c_te.stream_writer(fte) as wte:
            for obj in docs:
                d = domain_of(obj.get("url", ""))
                line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

                if d in test_set:
                    wte.write(line); counts["test"] += 1
                elif d in val_set:
                    wva.write(line); counts["val"] += 1
                else:
                    wtr.write(line); counts["train"] += 1

    print("Domains:", n_dom, "| test dom:", len(test_set), "| val dom:", len(val_set))
    print("Docs:", counts)
    print("Saved:", train_out, val_out, test_out)

if __name__ == "__main__":
    main()
