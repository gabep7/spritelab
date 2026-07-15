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
from diffusers import LCMScheduler, StableDiffusionXLPipeline
from PIL import Image

_peft_import_utils.is_torchao_available = lambda: False
_peft_torchao.is_torchao_available = lambda: False

print(f"torch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PIXEL_LORA_ID = "nerijs/pixel-art-xl"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdxl"
OUTPUT_DIR = Path("/kaggle/working/sdxl_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(LCM_LORA_ID, adapter_name="lcm")
pipe.load_lora_weights(
    PIXEL_LORA_ID,
    weight_name="pixel-art-xl.safetensors",
    adapter_name="pixel",
)
pipe.set_adapters(["lcm", "pixel"], adapter_weights=[1.0, 1.2])
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
print("SDXL, LCM, and Pixel Art XL loaded")

PROMPTS = [
    (
        "black_dragon_blue_fire",
        "pixel art, a black dragon breathing bright blue fire, full body, side view, isolated game sprite, plain white background",
    ),
    (
        "fire_mage",
        "pixel art, a fire mage holding a wooden staff, full body, isolated game sprite, plain white background",
    ),
    (
        "armored_knight",
        "pixel art, an armored knight holding a steel sword, full body, isolated game sprite, plain white background",
    ),
    (
        "female_archer",
        "pixel art, a female archer drawing a longbow, full body, isolated game sprite, plain white background",
    ),
    (
        "green_slime",
        "pixel art, a cute green slime monster, isolated game sprite, plain white background",
    ),
    (
        "red_robot",
        "pixel art, a red armored robot with glowing eyes, full body, isolated game sprite, plain white background",
    ),
    (
        "treasure_chest",
        "pixel art, a wooden treasure chest with gold trim, isolated game sprite, plain white background",
    ),
    (
        "blue_sword",
        "pixel art, a glowing blue crystal sword, isolated game sprite, plain white background",
    ),
    (
        "yellow_airship",
        "pixel art, a small yellow fantasy airship, side view, isolated game sprite, plain white background",
    ),
    (
        "castle_tower",
        "pixel art, a stone castle tower with a red banner, isolated game sprite, plain white background",
    ),
    (
        "purple_potion",
        "pixel art, a glass bottle filled with purple magic potion, isolated game sprite, plain white background",
    ),
    (
        "brown_dog",
        "pixel art, a small brown dog with a red collar, full body, side view, isolated game sprite, plain white background",
    ),
]
SEEDS = [42, 31415]
NEGATIVE_PROMPT = "3d render, realistic, photograph, text, watermark, scenery, complex background, cropped"
manifest = []
previews = []

for prompt_index, (name, prompt) in enumerate(PROMPTS):
    for seed in SEEDS:
        print(f"Generating {name}, seed {seed}")
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=1024,
            width=1024,
            num_inference_steps=8,
            guidance_scale=1.5,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
        stem = f"{prompt_index:02d}_{name}_seed_{seed}"
        raw_path = OUTPUT_DIR / f"{stem}_raw.png"
        pixel_path = OUTPUT_DIR / f"{stem}.png"
        image.save(raw_path)
        pixelated = image.resize((128, 128), Image.Resampling.NEAREST)
        pixelated = pixelated.resize((512, 512), Image.Resampling.NEAREST)
        pixelated.save(pixel_path)
        previews.append(pixelated.resize((256, 256), Image.Resampling.NEAREST))
        manifest.append(
            {
                "name": name,
                "prompt": prompt,
                "seed": seed,
                "raw": raw_path.name,
                "pixel": pixel_path.name,
            }
        )

with open(OUTPUT_DIR / "manifest.json", "w") as manifest_file:
    json.dump(manifest, manifest_file, indent=2)

grid_columns = 4
grid_rows = (len(previews) + grid_columns - 1) // grid_columns
grid = Image.new("RGB", (grid_columns * 256, grid_rows * 256), "white")
for index, preview in enumerate(previews):
    grid.paste(preview, ((index % grid_columns) * 256, (index // grid_columns) * 256))
grid.save("/kaggle/working/sdxl_benchmark_grid.png")

print(f"Saved {len(manifest)} outputs and sdxl_benchmark_grid.png")
print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
