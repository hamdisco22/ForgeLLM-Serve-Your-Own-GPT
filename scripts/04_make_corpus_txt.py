import io
import json
from pathlib import Path
import zstandard as zstd

IN_FILE = Path("data") / "clean" / "train.jsonl.zst"
OUT_TXT = Path("tokenizer") / "corpus.txt"

def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {IN_FILE}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    dctx = zstd.ZstdDecompressor()

    with open(IN_FILE, "rb") as fh, open(OUT_TXT, "w", encoding="utf-8") as out:
        with dctx.stream_reader(fh) as reader:
            # Convert decompressed bytes -> text lines
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.write(obj["text"].replace("\n", " ") + "\n")

    print(f"Wrote {OUT_TXT}")

if __name__ == "__main__":
    main()
