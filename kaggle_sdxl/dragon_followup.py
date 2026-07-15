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

OUTPUT_DIR = Path("/kaggle/working/sdxl_dragon_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

PROMPT = (
    "pixel art, full-body black dragon in side profile facing right, mouth wide open, "
    "actively exhaling a long visible stream of bright cyan-blue flames from its mouth, "
    "blue fire breath attack extending to the right, isolated game sprite, plain white background"
)
NEGATIVE = (
    "3d render, realistic, photograph, text, watermark, scenery, complex background, cropped, "
    "closed mouth, blue wings, blue body, no fire"
)
images = []
for seed in [42, 31415, 271828, 8675309]:
    print(f"Generating dragon, seed {seed}")
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=7.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]
    image.save(OUTPUT_DIR / f"dragon_seed_{seed}_raw.png")
    pixelated = image.resize((128, 128), Image.Resampling.NEAREST)
    pixelated = pixelated.resize((512, 512), Image.Resampling.NEAREST)
    pixelated.save(OUTPUT_DIR / f"dragon_seed_{seed}.png")
    images.append(pixelated)

comparison = Image.new("RGB", (1024, 1024), "white")
for index, image in enumerate(images):
    comparison.paste(image, ((index % 2) * 512, (index // 2) * 512))
comparison.save("/kaggle/working/sdxl_dragon_test_grid.png")
print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
