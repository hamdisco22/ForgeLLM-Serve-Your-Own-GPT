import urllib.request
from pathlib import Path
from tqdm import tqdm

IN_PATHS = Path("data") / "wet_sample_paths.txt"
OUT_DIR = Path("data") / "raw_wet"
BASE = "https://data.commoncrawl.org/"

def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return

    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get("Content-Length", "0"))
        chunk = 1024 * 1024
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                f.write(b)
                pbar.update(len(b))

def main():
    paths = [p.strip() for p in IN_PATHS.read_text(encoding="utf-8").splitlines() if p.strip()]
    for rel in paths:
        url = BASE + rel
        dest = OUT_DIR / Path(rel).name
        print("GET", url)
        download(url, dest)

    print("Done. Files in:", OUT_DIR)

if __name__ == "__main__":
    main()
