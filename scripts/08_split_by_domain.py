import io, json
from pathlib import Path
from urllib.parse import urlparse
import zstandard as zstd

IN_FILE = Path("data/clean/train_dedup_near.jsonl.zst")  # from step 1
OUT_DIR = Path("data/clean/splits")

TRAIN_OUT = OUT_DIR / "train.jsonl.zst"
VAL_OUT   = OUT_DIR / "val.jsonl.zst"
TEST_OUT  = OUT_DIR / "test.jsonl.zst"

# percentages
VAL_PCT = 1   # 1%
TEST_PCT = 1  # 1%

def iter_jsonl_zst(path: Path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            ts = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in ts:
                line = line.strip()
                if line:
                    yield json.loads(line)

def bucket_for_domain(domain: str):
    # deterministic split using hash(domain)
    h = (hash(domain) & 0xFFFFFFFF) % 100
    if h < TEST_PCT:
        return "test"
    if h < TEST_PCT + VAL_PCT:
        return "val"
    return "train"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=10)

    counts = {"train": 0, "val": 0, "test": 0}

    with open(TRAIN_OUT, "wb") as ftr, open(VAL_OUT, "wb") as fva, open(TEST_OUT, "wb") as fte:
        with cctx.stream_writer(ftr) as wtr, cctx.stream_writer(fva) as wva, cctx.stream_writer(fte) as wte:
            for obj in iter_jsonl_zst(IN_FILE):
                url = obj.get("url", "")
                domain = (urlparse(url).netloc or "unknown").lower()
                split = bucket_for_domain(domain)

                line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
                if split == "train":
                    wtr.write(line)
                elif split == "val":
                    wva.write(line)
                else:
                    wte.write(line)
                counts[split] += 1

    print("Saved splits:")
    print(" train:", TRAIN_OUT, counts["train"])
    print(" val  :", VAL_OUT, counts["val"])
    print(" test :", TEST_OUT, counts["test"])

if __name__ == "__main__":
    main()
