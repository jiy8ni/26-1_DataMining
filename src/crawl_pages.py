"""
Crawl the unique product pages (resolved_url) referenced in trial_items.csv to
collect page body text and product images for semantic-feature engineering.

Output:
    data/raw/crawl/pages.parquet
        resolved_url, fetch_status, page_text, image_files (list[str]), n_images
    data/raw/crawl/images/<url_hash>/*.jpg   downloaded images (top-N per page)

The crawl is cache-safe: a URL whose entry already exists in pages.parquet
(with fetch_status == 200) is skipped on re-run. Images already on disk are
re-used. Be polite: per-request delay, retries, timeout, custom User-Agent.

Usage:
    python src/crawl_pages.py --limit 5        # smoke test on first 5 URLs
    python src/crawl_pages.py                   # full crawl (all unique URLs)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

# --- paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
TRIAL_ITEMS = REPO_ROOT / "data" / "processed" / "trial_items.csv"
CRAWL_DIR = REPO_ROOT / "data" / "raw" / "crawl"
IMAGES_DIR = CRAWL_DIR / "images"
PAGES_PARQUET = CRAWL_DIR / "pages.parquet"

# --- crawl settings ----------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko,en;q=0.8",
}
REQUEST_TIMEOUT = 30          # seconds (product pages are heavy, 200-440 KB)
N_RETRIES = 3
RETRY_BACKOFF = 2.0          # seconds (multiplied by attempt)

# A shared session (keep-alive, cookies) markedly improves success on Korean
# e-commerce sites vs one-off requests.
_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)
POLITE_DELAY = 1.5           # seconds between page fetches
MAX_IMAGES_PER_PAGE = 20     # download at most N images per page
MIN_IMAGE_BYTES = 3000       # skip tiny icons / spacers
MIN_IMAGE_SIDE = 80          # skip thumbnails smaller than this (px)


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def fetch(url: str) -> requests.Response | None:
    """GET with retries + backoff via a shared session. Falls back to
    verify=False on SSL errors. Returns the Response or None on failure."""
    verify = True
    for attempt in range(1, N_RETRIES + 1):
        try:
            return _SESSION.get(url, timeout=REQUEST_TIMEOUT, verify=verify,
                                allow_redirects=True)
        except requests.exceptions.SSLError as exc:
            verify = False  # retry without cert verification
            if attempt == N_RETRIES:
                print(f"    [fetch failed:SSL] {url} :: {exc}")
                return None
            time.sleep(RETRY_BACKOFF * attempt)
        except requests.RequestException as exc:
            if attempt == N_RETRIES:
                print(f"    [fetch failed] {url} :: {exc}")
                return None
            time.sleep(RETRY_BACKOFF * attempt)
    return None


def extract_text(soup: BeautifulSoup) -> str:
    """Visible body text with script/style/noscript stripped."""
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # collapse runs of whitespace
    return " ".join(text.split())


def collect_image_urls(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Absolute image URLs from <img src> / common lazy-load attributes."""
    urls: list[str] = []
    seen: set[str] = set()
    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-original")
            or img.get("data-lazy-src")
        )
        if not src:
            continue
        src = src.strip()
        if src.startswith("data:"):
            continue
        absolute = urljoin(base_url, src)
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls


def download_images(image_urls: list[str], dest_dir: Path) -> list[str]:
    """Download up to MAX_IMAGES_PER_PAGE valid images. Returns saved paths
    (relative to REPO_ROOT). Cache-safe: existing files are reused."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for src in image_urls:
        if len(saved) >= MAX_IMAGES_PER_PAGE:
            break
        ext = Path(urlparse(src).path).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ""}:
            continue
        fname = url_hash(src) + (ext if ext else ".jpg")
        fpath = dest_dir / fname
        if fpath.exists() and fpath.stat().st_size >= MIN_IMAGE_BYTES:
            saved.append(str(fpath.relative_to(REPO_ROOT)))
            continue
        resp = fetch(src)
        if resp is None or resp.status_code != 200:
            continue
        content = resp.content
        if len(content) < MIN_IMAGE_BYTES:
            continue
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            continue
        if min(img.size) < MIN_IMAGE_SIDE:
            continue
        try:
            img.save(fpath, format="JPEG", quality=85)
        except Exception:
            continue
        saved.append(str(fpath.relative_to(REPO_ROOT)))
    return saved


def load_existing() -> dict[str, dict]:
    """Existing successful crawl rows keyed by resolved_url (for cache-skip)."""
    if not PAGES_PARQUET.exists():
        return {}
    df = pd.read_parquet(PAGES_PARQUET)
    return {
        row["resolved_url"]: row.to_dict()
        for _, row in df.iterrows()
        if int(row["fetch_status"]) == 200
    }


def crawl_one(url: str) -> dict:
    resp = fetch(url)
    if resp is None:
        return {"resolved_url": url, "fetch_status": 0,
                "page_text": "", "image_files": [], "n_images": 0}
    status = resp.status_code
    if status != 200:
        return {"resolved_url": url, "fetch_status": status,
                "page_text": "", "image_files": [], "n_images": 0}
    resp.encoding = resp.apparent_encoding or resp.encoding
    soup = BeautifulSoup(resp.text, "lxml")
    text = extract_text(soup)
    image_urls = collect_image_urls(soup, base_url=resp.url)
    image_files = download_images(image_urls, IMAGES_DIR / url_hash(url))
    return {
        "resolved_url": url,
        "fetch_status": status,
        "page_text": text,
        "image_files": image_files,
        "n_images": len(image_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Crawl only the first N unique URLs (smoke test).")
    parser.add_argument("--force", action="store_true",
                        help="Re-crawl even if cached successfully.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt only previously non-200 URLs, updating "
                             "their rows in pages.parquet in place.")
    args = parser.parse_args()

    CRAWL_DIR.mkdir(parents=True, exist_ok=True)

    # --- retry-failed mode: re-crawl only the non-200 rows ------------------
    if args.retry_failed:
        if not PAGES_PARQUET.exists():
            print("No pages.parquet yet — run a normal crawl first.")
            return
        existing = pd.read_parquet(PAGES_PARQUET)
        by_url = {r["resolved_url"]: r.to_dict() for _, r in existing.iterrows()}
        failed = [u for u, r in by_url.items() if int(r["fetch_status"]) != 200]
        print(f"Retrying {len(failed)} previously-failed URLs ...")
        recovered = 0
        for i, url in enumerate(failed, 1):
            print(f"[{i}/{len(failed)}] {url}")
            row = crawl_one(url)
            print(f"    status={row['fetch_status']} "
                  f"text_len={len(row['page_text'])} images={row['n_images']}")
            if row["fetch_status"] == 200:
                recovered += 1
            by_url[url] = row
            pd.DataFrame(list(by_url.values())).to_parquet(PAGES_PARQUET, index=False)
            time.sleep(POLITE_DELAY)
        df = pd.DataFrame(list(by_url.values()))
        df.to_parquet(PAGES_PARQUET, index=False)
        ok = int((df["fetch_status"] == 200).sum())
        print(f"\n=== retry summary === recovered {recovered}/{len(failed)}  "
              f"| total ok now: {ok}/{len(df)}")
        return

    items = pd.read_csv(TRIAL_ITEMS)
    urls = items["resolved_url"].dropna().unique().tolist()
    if args.limit is not None:
        urls = urls[: args.limit]
    print(f"Unique URLs to consider: {len(urls)}")

    cached = {} if args.force else load_existing()
    print(f"Already cached (status 200): {len(cached)}")

    results: list[dict] = list(cached.values())
    todo = [u for u in urls if u not in cached]
    print(f"To crawl this run: {len(todo)}")

    for i, url in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {url}")
        row = crawl_one(url)
        print(f"    status={row['fetch_status']} "
              f"text_len={len(row['page_text'])} images={row['n_images']}")
        results.append(row)
        # persist incrementally so an interrupted run keeps progress
        pd.DataFrame(results).to_parquet(PAGES_PARQUET, index=False)
        time.sleep(POLITE_DELAY)

    df = pd.DataFrame(results)
    df.to_parquet(PAGES_PARQUET, index=False)

    ok = int((df["fetch_status"] == 200).sum())
    print("\n=== crawl summary ===")
    print(f"rows: {len(df)}  ok(200): {ok}  failed: {len(df) - ok}")
    if ok:
        ok_df = df[df["fetch_status"] == 200]
        print(f"avg text length: {ok_df['page_text'].str.len().mean():.0f}")
        print(f"avg images/page: {ok_df['n_images'].mean():.1f}")
    print(f"saved -> {PAGES_PARQUET.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
