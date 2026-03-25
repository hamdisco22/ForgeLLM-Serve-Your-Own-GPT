import json
import re
from pathlib import Path
import xxhash
import zstandard as zstd
from tqdm import tqdm
from langdetect import detect, LangDetectException

IN_DIR = Path("data") / "extracted"
OUT_FILE = Path("data") / "clean" / "train.jsonl.zst"

# Simple quality filters
MIN_CHARS = 400
MAX_CHARS = 20_000
ALPHA_RATIO_MIN = 0.55
KEEP_LANGS = {"en"}  # change to {"en","fr","ar"} later if you want

_ws = re.compile(r"\s+")
_rep = re.compile(r"(.)\1{8,}")  # e.g., "!!!!!!!!" or "aaaaaaaaa"

def normalize(text: str) -> str:
    text = text.replace("\x00", "")
    text = _ws.sub(" ", text).strip()
    return text

def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    a = sum(ch.isalpha() for ch in text)
    return a / max(1, len(text))

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(IN_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit("No extracted jsonl found. Run 02_extract_wet_to_jsonl.py first.")

    seen = set()
    kept = 0
    read = 0

    cctx = zstd.ZstdCompressor(level=10)
    with open(OUT_FILE, "wb") as fbin, cctx.stream_writer(fbin) as zw:
        for fp in files:
            for line in tqdm(fp.open("r", encoding="utf-8"), desc=fp.name):
                read += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                url = obj.get("url", "")
                text = normalize(obj.get("text", ""))

                if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
                    continue
                if _rep.search(text):
                    continue
                if alpha_ratio(text) < ALPHA_RATIO_MIN:
                    continue

                # language id (cheap but useful)
                try:
                    lang = detect(text[:2000])
                except LangDetectException:
                    continue
                if lang not in KEEP_LANGS:
                    continue

                h = xxhash.xxh3_64_hexdigest(text)
                if h in seen:
                    continue
                seen.add(h)

                out = {"url": url, "lang": lang, "text": text}
                zw.write((json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8"))
                kept += 1

    print(f"Read={read:,} Kept={kept:,} Saved={OUT_FILE}")

if __name__ == "__main__":
    main()
