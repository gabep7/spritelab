import json
import subprocess
import sys
from pathlib import Path

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "diffusers==0.37.0",
        "transformers",
        "accelerate",
        "peft",
        "safetensors",
        "torchao>=0.16.0",
    ],
    check=True,
)

import torch
import peft.import_utils as _peft_import_utils
import peft.tuners.lora.torchao as _peft_torchao
from diffusers import StableDiffusionXLPipeline
from PIL import Image

_peft_import_utils.is_torchao_available = lambda: False
_peft_torchao.is_torchao_available = lambda: False

OUTPUT_DIR = Path("/kaggle/working/spritelab_showcase")
LOGO_DIR = OUTPUT_DIR / "logo_candidates"
EXAMPLE_DIR = OUTPUT_DIR / "examples"
LOGO_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe.load_lora_weights(
    "nerijs/pixel-art-xl",
    weight_name="pixel-art-xl.safetensors",
    adapter_name="pixel",
)
pipe.set_adapters("pixel", adapter_weights=1.2)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

NEGATIVE = (
    "3d render, realistic, photograph, blurry, smooth shading, text, words, letters, watermark, "
    "signature, scenery, complex background, cropped, multiple objects"
)
LOGO_PROMPT = (
    "pixel art game studio logo icon, one glass laboratory flask containing a tiny glowing blue "
    "sprite creature, orange stopper, cyan blue and warm orange color palette, bold readable "
    "silhouette, centered minimal 16-bit emblem, isolated on pure white background, no text"
)
EXAMPLE_PROMPTS = [
    (
        "black_dragon_blue_fire",
        "pixel art, full-body black dragon in side profile facing right, mouth open and breathing a long visible stream of bright cyan-blue fire to the right, isolated game sprite, plain white background",
    ),
    (
        "skeleton_knight",
        "pixel art, full-body skeleton knight in cracked dark armor holding a glowing red sword and round shield, isolated game sprite, plain white background",
    ),
    (
        "forest_witch",
        "pixel art, full-body forest witch with a moss green pointed hat holding a glowing mushroom staff, isolated game sprite, plain white background",
    ),
    (
        "retro_robot",
        "pixel art, full-body red retro robot with teal screen face and mechanical arms, isolated game sprite, plain white background",
    ),
    (
        "blue_slime",
        "pixel art, cute translucent blue slime monster with a tiny gold crown, isolated game sprite, plain white background",
    ),
    (
        "fantasy_airship",
        "pixel art, small brass fantasy airship with a blue balloon and side propellers, side view, isolated game sprite, plain white background",
    ),
    (
        "magic_potion",
        "pixel art, ornate glass potion bottle filled with glowing purple liquid and tiny stars, isolated game sprite, plain white background",
    ),
    (
        "treasure_chest",
        "pixel art, open wooden treasure chest with gold trim overflowing with colorful gems, isolated game sprite, plain white background",
    ),
]


def generate(prompt, seed):
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=7.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]


def pixelate(image):
    small = image.resize((128, 128), Image.Resampling.NEAREST)
    return small.resize((512, 512), Image.Resampling.NEAREST)


def remove_white_background(image):
    rgba = image.convert("RGBA")
    pixels = []
    for red, green, blue, alpha in rgba.getdata():
        if red >= 245 and green >= 245 and blue >= 245:
            pixels.append((255, 255, 255, 0))
        else:
            pixels.append((red, green, blue, alpha))
    rgba.putdata(pixels)
    return rgba


manifest = {"logo_candidates": [], "examples": []}
logo_previews = []
showcase_previews = []

for seed in [42, 31415, 271828, 8675309]:
    print(f"Generating logo candidate, seed {seed}")
    image = generate(LOGO_PROMPT, seed)
    image.save(LOGO_DIR / f"logo_seed_{seed}_raw.png")
    processed = pixelate(image)
    processed.save(LOGO_DIR / f"logo_seed_{seed}.png")
    remove_white_background(processed).save(LOGO_DIR / f"logo_seed_{seed}_transparent.png")
    logo_previews.append(processed)
    showcase_previews.append(processed)
    manifest["logo_candidates"].append({"seed": seed, "prompt": LOGO_PROMPT})

for name, prompt in EXAMPLE_PROMPTS:
    for seed in [42, 31415]:
        print(f"Generating {name}, seed {seed}")
        image = generate(prompt, seed)
        image.save(EXAMPLE_DIR / f"{name}_seed_{seed}_raw.png")
        processed = pixelate(image)
        processed.save(EXAMPLE_DIR / f"{name}_seed_{seed}.png")
        showcase_previews.append(processed)
        manifest["examples"].append({"name": name, "seed": seed, "prompt": prompt})

logo_grid = Image.new("RGB", (1024, 256), "white")
for index, image in enumerate(logo_previews):
    logo_grid.paste(image.resize((256, 256), Image.Resampling.NEAREST), (index * 256, 0))
logo_grid.save("/kaggle/working/spritelab_logo_candidates.png")

columns = 4
rows = (len(showcase_previews) + columns - 1) // columns
showcase_grid = Image.new("RGB", (columns * 256, rows * 256), "white")
for index, image in enumerate(showcase_previews):
    x = index % columns * 256
    y = index // columns * 256
    showcase_grid.paste(image.resize((256, 256), Image.Resampling.NEAREST), (x, y))
showcase_grid.save("/kaggle/working/spritelab_showcase_grid.png")

with open(OUTPUT_DIR / "manifest.json", "w") as manifest_file:
    json.dump(manifest, manifest_file, indent=2)

print(f"Saved {len(showcase_previews)} generated images")
print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
