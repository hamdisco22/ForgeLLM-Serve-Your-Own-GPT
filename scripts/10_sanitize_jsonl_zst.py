import io
import json
import argparse
from pathlib import Path

import zstandard as zstd


def sanitize_line(line: str) -> str:
    # remove null bytes and weird whitespace; keep JSON escape sequences intact
    return line.replace("\x00", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True)
    ap.add_argument("--outfile", type=str, required=True)
    ap.add_argument("--logfile", type=str, default=None)
    ap.add_argument("--max_errors", type=int, default=20)
    args = ap.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    logfile = Path(args.logfile) if args.logfile else outfile.with_suffix(".bad.log")

    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor(level=10)

    kept = 0
    bad = 0

    with open(infile, "rb") as fin, open(outfile, "wb") as fout, open(logfile, "w", encoding="utf-8") as flog:
        with dctx.stream_reader(fin) as reader, cctx.stream_writer(fout) as writer:
            ts = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

            for lineno, raw in enumerate(ts, start=1):
                line = sanitize_line(raw)
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    # minimal schema check
                    if not isinstance(obj, dict) or "text" not in obj:
                        raise ValueError("missing 'text'")
                except Exception as e:
                    bad += 1
                    if bad <= args.max_errors:
                        # log a short preview (don’t dump entire huge lines)
                        preview = line[:300].replace("\n", "\\n")
                        flog.write(f"[line {lineno}] {type(e).__name__}: {e}\n{preview}\n\n")
                    continue

                writer.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                kept += 1

    print(f"Sanitize done: kept={kept:,} bad={bad:,}")
    print(f"Output: {outfile}")
    print(f"Log   : {logfile}")


if __name__ == "__main__":
    main()
