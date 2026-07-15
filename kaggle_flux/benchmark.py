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
        "git+https://github.com/huggingface/diffusers.git",
        "transformers",
        "accelerate",
        "peft",
        "safetensors",
        "sentencepiece",
        "torchao>=0.16.0",
    ],
    check=True,
)

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import peft.import_utils as _peft_import_utils
import peft.tuners.lora.torchao as _peft_torchao

_peft_import_utils.is_torchao_available = lambda: False
_peft_torchao.is_torchao_available = lambda: False

print(f"torch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
LORA_ID = "Limbicnation/pixel-art-lora"
OUTPUT_DIR = Path("/kaggle/working/flux_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipe = Flux2KleinPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.load_lora_weights(LORA_ID, weight_name="pytorch_lora_weights.safetensors")
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
print("FLUX.2 Klein 4B and pixel-art LoRA loaded")

PROMPTS = [
    (
        "black_dragon_blue_fire",
        "pixel art sprite, a black dragon breathing bright blue fire, full body, side view, game asset, isolated on transparent background",
    ),
    (
        "fire_mage",
        "pixel art sprite, a fire mage holding a wooden staff, full body, game asset, isolated on transparent background",
    ),
    (
        "armored_knight",
        "pixel art sprite, an armored knight holding a steel sword, full body, game asset, isolated on transparent background",
    ),
    (
        "female_archer",
        "pixel art sprite, a female archer drawing a longbow, full body, game asset, isolated on transparent background",
    ),
    (
        "green_slime",
        "pixel art sprite, a cute green slime monster, game asset, isolated on transparent background",
    ),
    (
        "red_robot",
        "pixel art sprite, a red armored robot with glowing eyes, full body, game asset, isolated on transparent background",
    ),
    (
        "treasure_chest",
        "pixel art sprite, a wooden treasure chest with gold trim, game asset, isolated on transparent background",
    ),
    (
        "blue_sword",
        "pixel art sprite, a glowing blue crystal sword, game asset, isolated on transparent background",
    ),
    (
        "yellow_airship",
        "pixel art sprite, a small yellow fantasy airship, side view, game asset, isolated on transparent background",
    ),
    (
        "castle_tower",
        "pixel art sprite, a stone castle tower with a red banner, game asset, isolated on transparent background",
    ),
    (
        "purple_potion",
        "pixel art sprite, a glass bottle filled with purple magic potion, game asset, isolated on transparent background",
    ),
    (
        "brown_dog",
        "pixel art sprite, a small brown dog with a red collar, full body, side view, game asset, isolated on transparent background",
    ),
]
SEEDS = [42, 31415]
manifest = []
previews = []

for prompt_index, (name, prompt) in enumerate(PROMPTS):
    for seed in SEEDS:
        print(f"Generating {name}, seed {seed}")
        image = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=1.0,
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
grid.save("/kaggle/working/flux_benchmark_grid.png")

print(f"Saved {len(manifest)} outputs and flux_benchmark_grid.png")
print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
