"""
Scrape sprite sheets from The Spriter's Resource.

Pulls from the GBA section. Saves raw sprite sheets to dataset/raw/.
"""

import re
import time
import json
import urllib.request
import ssl
from pathlib import Path
from sprite_rules import include_sheet, sheet_subject

BASE_URL = "https://www.spriters-resource.com"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "raw"
META_PATH = ROOT / "dataset" / "raw" / "_meta.json"
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries}: {e}")
            time.sleep(2)
    return None


def fetch_bytes(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
                return resp.read()
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries}: {e}")
            time.sleep(2)
    return None


def parse_game_links_from_page(html):
    seen = set()
    games = []
    for m in re.finditer(r'href="/game_boy_advance/([a-z0-9_\-]+)/"', html):
        slug = m.group(1)
        if slug not in seen and len(slug) > 1:
            seen.add(slug)
            games.append(slug)
    return games


def fetch_all_game_links(letters):
    all_games = []
    seen = set()
    for letter in letters:
        url = f"{BASE_URL}/game_boy_advance/{letter}/"
        html = fetch(url)
        if not html:
            continue
        for slug in parse_game_links_from_page(html):
            if slug not in seen:
                seen.add(slug)
                all_games.append(slug)
        print(f"  {letter}: {len(all_games)} games total")
        time.sleep(0.3)
    return all_games


def parse_sheet_links(html):
    seen = set()
    sheets = []
    for m in re.finditer(r'href="/game_boy_advance/[a-z0-9_\-]+/asset/(\d+)/"', html):
        sid = m.group(1)
        if sid not in seen:
            seen.add(sid)
            sheets.append(sid)
    return sheets


def parse_sheet_download_url(html):
    pattern = re.compile(r'(?:src|href)="(/media/assets/[^"]+\.png[^"]*)"', re.IGNORECASE)
    matches = pattern.findall(html)
    if matches:
        return BASE_URL + matches[0]
    return None


def parse_sheet_name(html):
    m = re.search(r"<title>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        title = re.sub(r"\s*-\s*The Spriter.*$", "", title)
        return title
    return "unknown"


def main(max_games=20, max_sheets_per_game=30, requested_games=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["0"]
    print("Fetching GBA game list...")
    available_games = fetch_all_game_links(letters)
    print(f"Found {len(available_games)} games total")
    if requested_games:
        available = set(available_games)
        missing = [game for game in requested_games if game not in available]
        if missing:
            raise ValueError(f"Unknown GBA game slugs: {', '.join(missing)}")
        games = requested_games
    else:
        games = available_games[:max_games] if max_games else available_games
    print(f"Scraping {len(games)} games (max {max_sheets_per_game} sheets each)")

    meta = []
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
    existing_sheet_ids = {str(item["sheet_id"]) for item in meta}

    for i, slug in enumerate(games):
        game_dir = RAW_DIR / slug
        game_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i + 1}/{len(games)}] {slug}")
        game_url = f"{BASE_URL}/game_boy_advance/{slug}/"
        game_html = fetch(game_url)
        if not game_html:
            continue
        time.sleep(1)
        sheet_ids = parse_sheet_links(game_html)[:max_sheets_per_game]
        print(f"  found {len(sheet_ids)} sheets")
        for j, sid in enumerate(sheet_ids):
            if sid in existing_sheet_ids:
                print(f"  [{j + 1}/{len(sheet_ids)}] cached sheet {sid}")
                continue
            sheet_url = f"{BASE_URL}/game_boy_advance/{slug}/asset/{sid}/"
            sheet_html = fetch(sheet_url)
            if not sheet_html:
                continue
            time.sleep(0.5)
            name = parse_sheet_name(sheet_html)
            subject = sheet_subject(name)
            if not include_sheet(subject):
                print(f"  [{j + 1}/{len(sheet_ids)}] skipped non-sprite sheet ({subject})")
                continue
            img_url = parse_sheet_download_url(sheet_html)
            if not img_url:
                continue
            img_data = fetch_bytes(img_url)
            if not img_data:
                continue
            time.sleep(0.5)
            safe = re.sub(r"[^a-zA-Z0-9\-_]", "_", name)[:60]
            file_name = f"{safe}_{sid}.png"
            destination = game_dir / file_name
            destination.write_bytes(img_data)
            meta.append(
                {
                    "game": slug,
                    "sheet_name": name,
                    "sheet_id": sid,
                    "file": str(destination.relative_to(ROOT)),
                    "source_url": img_url,
                }
            )
            existing_sheet_ids.add(sid)
            print(f"  [{j + 1}/{len(sheet_ids)}] saved {file_name} ({len(img_data)} bytes)")
        META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\nDone. {len(meta)} sheets saved to {RAW_DIR}")


if __name__ == "__main__":
    import sys

    maximum_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    maximum_sheets = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    selected_games = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    main(maximum_games, maximum_sheets, selected_games)