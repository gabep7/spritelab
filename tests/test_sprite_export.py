from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sprite_export import extract_sprite, save_sprite_outputs


def _make_white_sprite(size=256, box=(80, 60, 180, 200), color=(30, 90, 200)):
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, fill=color)
    draw.ellipse((100, 40, 160, 100), fill=(240, 90, 40))
    return image


def test_extract_sprite_centers_and_makes_transparent():
    image = _make_white_sprite()
    sprite = extract_sprite(image, size=128)
    assert sprite.size == (128, 128)
    assert sprite.mode == "RGBA"

    alpha = sprite.getchannel("A")
    assert alpha.getbbox() is not None
    # Corners of the export canvas should stay transparent.
    for point in [(0, 0), (127, 0), (0, 127), (127, 127)]:
        assert sprite.getpixel(point)[3] == 0
    # Some opaque foreground must exist.
    assert max(alpha.getdata()) == 255


def test_extract_sprite_rejects_blank_image():
    blank = Image.new("RGB", (256, 256), (255, 255, 255))
    try:
        extract_sprite(blank, size=64)
        assert False, "expected RuntimeError for blank image"
    except RuntimeError as error:
        assert "no foreground" in str(error)


def test_save_sprite_outputs_writes_three_files(tmp_path):
    image = _make_white_sprite()
    output = tmp_path / "latest.png"
    paths = save_sprite_outputs(image, output, size=96)
    assert paths["sprite"].is_file()
    assert paths["preview"].is_file()
    assert paths["raw"].is_file()
    assert Image.open(paths["sprite"]).size == (96, 96)
    assert Image.open(paths["preview"]).size == (512, 512)
    assert Image.open(paths["raw"]).size == (256, 256)
