import gzip
import json
from pathlib import Path
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

RAW_DIR = Path("data") / "raw_wet"
OUT_DIR = Path("data") / "extracted"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wet_files = sorted(RAW_DIR.glob("*.gz"))
    if not wet_files:
        raise SystemExit("No WET files found. Run 01_download_wet.py first.")

    for wet in wet_files:
        out = OUT_DIR / (wet.stem + ".jsonl")
        if out.exists() and out.stat().st_size > 0:
            print("Skip existing:", out)
            continue

        with gzip.open(wet, "rb") as stream, open(out, "w", encoding="utf-8") as f:
            n = 0
            for record in tqdm(ArchiveIterator(stream), desc=wet.name):
                if record.rec_type != "conversion":
                    continue

                url = record.rec_headers.get_header("WARC-Target-URI")
                if not url:
                    continue

                try:
                    payload = record.content_stream().read()
                    text = payload.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue

                if not text:
                    continue

                obj = {"url": url, "text": text}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1

        print(f"Wrote {out} ({n} docs)")

if __name__ == "__main__":
    main()
