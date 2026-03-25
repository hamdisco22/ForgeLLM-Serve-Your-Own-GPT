import io, json, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import zstandard as zstd
import xxhash
from tqdm import tqdm

IN_FILE  = Path("data/clean/train.jsonl.zst")
OUT_FILE = Path("data/clean/train_dedup_near.jsonl.zst")

# tuning
PREFIX_BITS = 16          # bucket key (bigger -> fewer comparisons)
HAMM_MAX = 3              # consider near-dup if hamming <= 3
MAX_DOCS_PER_BUCKET = 500 # safety cap for huge buckets

_word = re.compile(r"[A-Za-zÀ-ÿ0-9_']+")
def tokenize(text: str):
    return _word.findall(text.lower())

def simhash64(tokens):
    # 64-dim signed accumulator
    v = np.zeros(64, dtype=np.int32)
    for t in tokens:
        h = xxhash.xxh3_64_intdigest(t)  # 64-bit
        for i in range(64):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out

def hamming64(a, b):
    return (a ^ b).bit_count()

def iter_jsonl_zst(path: Path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            ts = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in ts:
                line = line.strip()
                if line:
                    yield json.loads(line)

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1) first pass: compute hashes and bucket them
    buckets = defaultdict(list)  # prefix -> [(idx, hash)]
    docs = []                    # store minimal doc fields in memory (url, lang, text)
    for obj in tqdm(iter_jsonl_zst(IN_FILE), desc="Hashing"):
        text = obj.get("text", "")
        url  = obj.get("url", "")
        lang = obj.get("lang", "")
        toks = tokenize(text)
        if len(toks) < 50:
            continue
        h = simhash64(toks[:2000])  # cap tokens for speed
        idx = len(docs)
        docs.append((url, lang, text, h))
        prefix = h >> (64 - PREFIX_BITS)
        buckets[prefix].append(idx)

    keep = np.ones(len(docs), dtype=bool)

    # 2) within each bucket, drop near duplicates
    for prefix, idxs in tqdm(buckets.items(), desc="Dedup buckets"):
        if len(idxs) <= 1:
            continue
        if len(idxs) > MAX_DOCS_PER_BUCKET:
            idxs = idxs[:MAX_DOCS_PER_BUCKET]

        reps = []
        for i in idxs:
            if not keep[i]:
                continue
            hi = docs[i][3]
            is_dup = False
            for j in reps:
                hj = docs[j][3]
                if hamming64(hi, hj) <= HAMM_MAX:
                    is_dup = True
                    break
            if is_dup:
                keep[i] = False
            else:
                reps.append(i)

    # 3) write kept docs
    cctx = zstd.ZstdCompressor(level=10)
    kept = 0
    with open(OUT_FILE, "wb") as fh:
        with cctx.stream_writer(fh) as zw:
            for i, (url, lang, text, _) in enumerate(docs):
                if not keep[i]:
                    continue
                zw.write((json.dumps({"url": url, "lang": lang, "text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
                kept += 1

    print(f"Input docs: {len(docs):,}")
    print(f"Kept docs : {kept:,}")
    print(f"Saved     : {OUT_FILE}")

if __name__ == "__main__":
    main()
