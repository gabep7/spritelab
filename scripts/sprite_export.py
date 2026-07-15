from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def _color_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _sample_background_color(image):
    width, height = image.size
    samples = [
        image.getpixel((0, 0)),
        image.getpixel((width - 1, 0)),
        image.getpixel((0, height - 1)),
        image.getpixel((width - 1, height - 1)),
        image.getpixel((width // 2, 0)),
        image.getpixel((width // 2, height - 1)),
        image.getpixel((0, height // 2)),
        image.getpixel((width - 1, height // 2)),
    ]
    return tuple(sum(channel) // len(samples) for channel in zip(*samples))


def extract_sprite(image, size=128, padding_ratio=0.09, flood_thresh=48, fringe=28):
    """Crop a white-background generation into a centered transparent sprite."""
    source = image.convert("RGB")
    background = source.copy()
    marker = (254, 1, 253)
    corners = [
        (0, 0),
        (background.width - 1, 0),
        (0, background.height - 1),
        (background.width - 1, background.height - 1),
    ]
    for corner in corners:
        if background.getpixel(corner) != marker:
            ImageDraw.floodfill(background, corner, marker, thresh=flood_thresh)

    bg_color = _sample_background_color(source)
    rgba = source.convert("RGBA")
    alpha_values = []
    for pixel, filled in zip(source.getdata(), background.getdata()):
        if filled == marker:
            alpha_values.append(0)
            continue
        if _color_distance(pixel, bg_color) <= fringe and min(pixel) >= 220:
            alpha_values.append(0)
            continue
        alpha_values.append(255)

    alpha = Image.new("L", source.size)
    alpha.putdata(alpha_values)
    # Soft cleanup of single-pixel holes without eating solid silhouettes.
    alpha = alpha.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    rgba.putalpha(alpha)

    content_box = alpha.getbbox()
    if content_box is None:
        raise RuntimeError("Generated image contains no foreground")

    content = rgba.crop(content_box)
    content_limit = max(1, round(size * (1.0 - 2 * padding_ratio)))
    scale = min(content_limit / content.width, content_limit / content.height)
    resized = content.resize(
        (max(1, round(content.width * scale)), max(1, round(content.height * scale))),
        Image.Resampling.NEAREST,
    )
    sprite = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sprite.alpha_composite(
        resized,
        ((size - resized.width) // 2, (size - resized.height) // 2),
    )
    return sprite


def save_sprite_outputs(image, output, size=128):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(f"{output.stem}_raw.png")
    preview_output = output.with_name(f"{output.stem}_preview.png")
    image.save(raw_output)
    sprite = extract_sprite(image, size=size)
    sprite.save(output)
    sprite.resize((512, 512), Image.Resampling.NEAREST).save(preview_output)
    return {
        "sprite": output,
        "preview": preview_output,
        "raw": raw_output,
    }
