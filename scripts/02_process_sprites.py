"""
Process raw sprite sheets into individual sprites.
Crops sheets using background detection + flood fill.
Writes to dataset/sprites/ with metadata.jsonl for training.
"""

import hashlib
import json
from pathlib import Path

from PIL import Image
from sprite_rules import include_sheet, sheet_subject, subject_kind

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "raw"
SPRITES_DIR = ROOT / "dataset" / "sprites"

MIN_SIZE = 16
MAX_SIZE = 128
MAX_PER_SHEET = 32
BG_TOL = 30
MIN_FOREGROUND_RATIO = 0.05
MAX_FOREGROUND_RATIO = 0.85



def detect_bg(img):
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    pixels = []
    for x, y in [(0,0),(w-1,0),(0,h-1),(w-1,h-1),(w//2,0)]:
        pixels.append(img_rgb.getpixel((x, y)))
    counts = {}
    for p in pixels:
        counts[p] = counts.get(p, 0) + 1
    return max(counts, key=counts.get)


def is_bg(pixel, bg):
    return all(abs(pixel[i] - bg[i]) <= BG_TOL for i in range(3))


def find_bboxes(img, bg):
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    visited = [[False]*w for _ in range(h)]
    bboxes = []
    for y in range(h):
        for x in range(w):
            if visited[y][x]:
                continue
            pixel = img_rgb.getpixel((x, y))
            if is_bg(pixel, bg):
                visited[y][x] = True
                continue
            min_x, max_x, min_y, max_y = x, x, y, y
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if cx < 0 or cx >= w or cy < 0 or cy >= h:
                    continue
                if visited[cy][cx]:
                    continue
                cp = img_rgb.getpixel((cx, cy))
                if is_bg(cp, bg):
                    visited[cy][cx] = True
                    continue
                visited[cy][cx] = True
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                stack.extend([(cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)])
            bw, bh = max_x-min_x+1, max_y-min_y+1
            if MIN_SIZE <= bw <= MAX_SIZE and MIN_SIZE <= bh <= MAX_SIZE:
                bboxes.append((min_x, min_y, bw, bh))
    return bboxes




def evenly_spaced(items, limit):
    if len(items) <= limit:
        return items
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


def render_sprite(img, bbox, bg):
    x, y, w, h = bbox
    crop = img.crop((x, y, x + w, y + h)).convert("RGBA")
    backgrounds = [bg]
    corner_pixels = [
        crop.getpixel(position)
        for position in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))
    ]
    opaque_corner_colors = [pixel[:3] for pixel in corner_pixels if pixel[3] > 0]
    for candidate in opaque_corner_colors:
        matching_corners = sum(is_bg(color, candidate) for color in opaque_corner_colors)
        if matching_corners < 3:
            continue
        matching_area = sum(
            alpha > 0 and is_bg((red, green, blue), candidate)
            for red, green, blue, alpha in crop.getdata()
        )
        if matching_area / (w * h) >= 0.25:
            backgrounds.append(candidate)
            break

    cleaned = []
    for red, green, blue, alpha in crop.getdata():
        if alpha == 0 or any(is_bg((red, green, blue), color) for color in backgrounds):
            cleaned.append((255, 255, 255, 0))
        else:
            cleaned.append((red, green, blue, alpha))
    crop.putdata(cleaned)

    content_bbox = crop.getchannel("A").getbbox()
    if content_bbox is None:
        return None
    crop = crop.crop(content_bbox)
    w, h = crop.size
    if min(w, h) < MIN_SIZE or max(w, h) > MAX_SIZE:
        return None

    size = max(w, h)
    opaque_pixels = sum(alpha > 0 for alpha in crop.getchannel("A").getdata())
    if opaque_pixels / (w * h) >= 0.9:
        return None
    foreground_ratio = opaque_pixels / (size * size)
    if not MIN_FOREGROUND_RATIO <= foreground_ratio < MAX_FOREGROUND_RATIO:
        return None
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    canvas.alpha_composite(crop, ((size - w) // 2, (size - h) // 2))
    rgb = Image.new("RGB", canvas.size, "white")
    rgb.paste(canvas, mask=canvas.getchannel("A"))
    return rgb


def process_sheet(sheet_path, subject, game_name, seen_hashes):
    img = Image.open(sheet_path).convert("RGBA")
    bg = detect_bg(img)
    bboxes = find_bboxes(img, bg)
    candidates = []
    local_hashes = set()
    duplicate_count = 0

    for index, bbox in enumerate(bboxes):
        sprite = render_sprite(img, bbox, bg)
        if sprite is None:
            continue
        digest = hashlib.sha256(sprite.tobytes()).digest()
        if digest in local_hashes:
            duplicate_count += 1
            continue
        local_hashes.add(digest)
        candidates.append((index, sprite, digest))

    capped_count = max(0, len(candidates) - MAX_PER_SHEET)
    candidates = evenly_spaced(candidates, MAX_PER_SHEET)
    sprites = []
    global_duplicate_count = 0
    kind = subject_kind(subject)
    caption = f"pixel art sprite, {subject}, gba style, isolated on white background"

    for index, sprite, digest in candidates:
        if digest in seen_hashes:
            global_duplicate_count += 1
            continue
        seen_hashes.add(digest)
        file_name = f"{sheet_path.stem}_{index:03d}.png"
        sprite.save(SPRITES_DIR / file_name)
        sprites.append(
            {
                "file_name": file_name,
                "caption": caption,
                "game": game_name,
                "subject": subject,
                "kind": kind,
                "size": sprite.width,
            }
        )

    stats = {
        "components": len(bboxes),
        "local_duplicates": duplicate_count,
        "capped": capped_count,
        "global_duplicates": global_duplicate_count,
    }
    return sprites, stats


def main():
    SPRITES_DIR.mkdir(parents=True, exist_ok=True)
    for old_sprite in SPRITES_DIR.glob("*.png"):
        old_sprite.unlink()

    raw_metadata = json.loads((RAW_DIR / "_meta.json").read_text())
    metadata_by_file = {item["file"]: item for item in raw_metadata}
    game_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    print(f"Found {len(game_dirs)} game directories")

    all_sprites = []
    seen_hashes = set()
    skipped_sheets = 0
    for game_dir in game_dirs:
        game_name = game_dir.name.replace("_", " ").title()
        sheets = sorted(game_dir.glob("*.png"))
        if not sheets:
            continue
        print(f"\n{game_name}: {len(sheets)} sheets")
        for sheet in sheets:
            relative_path = str(sheet.relative_to(ROOT))
            metadata = metadata_by_file.get(relative_path)
            full_name = metadata["sheet_name"] if metadata else sheet.stem.replace("_", " ")
            subject = sheet_subject(full_name)
            if not include_sheet(subject):
                skipped_sheets += 1
                print(f"  {sheet.name}: skipped non-sprite sheet ({subject})")
                continue
            try:
                sprites, stats = process_sheet(sheet, subject, game_name, seen_hashes)
                all_sprites.extend(sprites)
                print(
                    f"  {sheet.name}: {len(sprites)} kept from {stats['components']} components "
                    f"({stats['local_duplicates']} duplicate, {stats['capped']} capped, "
                    f"{stats['global_duplicates']} cross-sheet duplicate)"
                )
            except Exception as error:
                print(f"  {sheet.name}: ERROR {error}")

    with open(SPRITES_DIR / "metadata.jsonl", "w") as metadata_file:
        for sprite in all_sprites:
            metadata_file.write(json.dumps(sprite) + "\n")
    print(
        f"\nDone. {len(all_sprites)} balanced sprites extracted to {SPRITES_DIR}; "
        f"{skipped_sheets} non-sprite sheets skipped"
    )


if __name__ == "__main__":
    main()