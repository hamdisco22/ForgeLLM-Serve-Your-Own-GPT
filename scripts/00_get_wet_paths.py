import argparse
import gzip
import io
import random
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--crawl", default="CC-MAIN-2026-04")
ap.add_argument("--n", type=int, default=30)
args = ap.parse_args()
CRAWL = args.crawl
N = args.n
OUT = Path("data") / "wet_sample_paths.txt"


def main():
    url = f"https://data.commoncrawl.org/crawl-data/{CRAWL}/wet.paths.gz"
    print("Downloading:", url)

    raw = urllib.request.urlopen(url, timeout=60).read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        lines = [ln.decode("utf-8").strip() for ln in gz if ln.strip()]

    # sample without loading the whole thing next time
    sample = random.sample(lines, k=min(N, len(lines)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(sample) + "\n", encoding="utf-8")

    print(f"Saved {len(sample)} paths to {OUT}")

if __name__ == "__main__":
    main()
