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
        "transformers==4.57.1",
        "accelerate==1.10.1",
        "peft==0.17.1",
        "safetensors==0.6.2",
        "torchao>=0.16.0",
    ],
    check=True,
)

import torch
import peft.import_utils as _peft_import_utils
import peft.tuners.lora.torchao as _peft_torchao
from diffusers import DPMSolverMultistepScheduler, LCMScheduler, StableDiffusionXLPipeline
from PIL import Image

from sprite_export import extract_sprite

_peft_import_utils.is_torchao_available = lambda: False
_peft_torchao.is_torchao_available = lambda: False

# quality: 30 steps DPM++ Karras. fast: 8 step LCM.
MODE = "quality"
EXPORT_SIZE = 128
SEEDS = [42, 31415]
OUTPUT_DIR = Path("/kaggle/working/sdxl_generate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEGATIVE = (
    "3d render, realistic, photograph, blurry, smooth shading, text, words, letters, "
    "watermark, signature, scenery, complex background, cropped, multiple objects, "
    "panel, sheet, collage"
)

# Edit this list for batch generation.
JOBS = [
    (
        "black_dragon_blue_fire",
        "pixel art, full-body black dragon in side profile facing right, mouth open, "
        "breathing a long visible stream of bright cyan-blue fire to the right, "
        "isolated game sprite, plain white background",
    ),
    (
        "skeleton_knight",
        "pixel art, full-body skeleton knight in cracked dark armor holding a glowing red sword "
        "and round shield, isolated game sprite, plain white background",
    ),
    (
        "forest_witch",
        "pixel art, full-body forest witch with a moss green pointed hat holding a glowing mushroom "
        "staff, isolated game sprite, plain white background",
    ),
    (
        "treasure_chest",
        "pixel art, wooden treasure chest with gold trim and a glowing gem lock, isolated game sprite, "
        "plain white background",
    ),
    (
        "magic_potion",
        "pixel art, glass potion bottle filled with glowing cyan liquid, isolated game sprite, "
        "plain white background",
    ),
    (
        "fantasy_airship",
        "pixel art, fantasy wooden airship with blue sails and brass fittings, isolated game sprite, "
        "plain white background",
    ),
]

print(f"torch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"mode={MODE} export={EXPORT_SIZE} jobs={len(JOBS)} seeds={SEEDS}")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
scheduler_config = pipe.scheduler.config
quality_scheduler = DPMSolverMultistepScheduler.from_config(
    scheduler_config,
    use_karras_sigmas=True,
)
fast_scheduler = LCMScheduler.from_config(scheduler_config)
pipe.load_lora_weights(
    "nerijs/pixel-art-xl",
    weight_name="pixel-art-xl.safetensors",
    adapter_name="pixel",
)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")

if MODE == "fast":
    pipe.scheduler = fast_scheduler
    pipe.set_adapters(["lcm", "pixel"], adapter_weights=[1.0, 1.2])
    steps = 8
    guidance = 1.5
else:
    pipe.scheduler = quality_scheduler
    pipe.set_adapters("pixel", adapter_weights=1.2)
    steps = 30
    guidance = 7.0

pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

manifest = []
previews = []
for job_name, prompt in JOBS:
    for seed in SEEDS:
        print(f"Generating {job_name}, seed {seed}")
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            height=1024,
            width=1024,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

        stem = f"{job_name}_seed_{seed}"
        raw_path = OUTPUT_DIR / f"{stem}_raw.png"
        sprite_path = OUTPUT_DIR / f"{stem}.png"
        image.save(raw_path)
        sprite = extract_sprite(image, size=EXPORT_SIZE)
        sprite.save(sprite_path)
        preview = sprite.resize((256, 256), Image.Resampling.NEAREST).convert("RGB")
        previews.append(preview)
        manifest.append(
            {
                "name": job_name,
                "seed": seed,
                "mode": MODE,
                "steps": steps,
                "guidance": guidance,
                "raw": str(raw_path.name),
                "sprite": str(sprite_path.name),
            }
        )

with open(OUTPUT_DIR / "manifest.json", "w") as manifest_file:
    json.dump(manifest, manifest_file, indent=2)

columns = 4
rows = (len(previews) + columns - 1) // columns
grid = Image.new("RGB", (columns * 256, max(1, rows) * 256), "white")
for index, preview in enumerate(previews):
    grid.paste(preview, ((index % columns) * 256, (index // columns) * 256))
grid_path = Path("/kaggle/working/sdxl_generate_grid.png")
grid.save(grid_path)

print(f"Saved {len(manifest)} sprites and {grid_path.name}")
print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
