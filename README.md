# Spritelab

Spritelab is an experimental pixel art sprite generator built around Stable Diffusion XL and the Pixel Art XL adapter. The selected workflow runs on a Kaggle Tesla T4 and generates isolated game assets from text prompts.

## Current model path

The selected pipeline is:

- Stable Diffusion XL Base 1.0
- Pixel Art XL
- Optional LCM adapter for eight step generation
- 1024 by 1024 generation
- Nearest neighbor downscaling to 128 by 128

The SDXL benchmark produced recognizable dragons, mages, knights, archers, slimes, robots, treasure chests, swords, airships, towers, potions, and dogs. It used less than 6 GB of GPU memory on a Kaggle T4.

The earlier SD 1.5 LoRA remains in the repository for reference, but it is not the selected model. Its prompt control and composition quality are not good enough.

## Kaggle generator

The current Kaggle script is in `kaggle_sdxl/dragon_followup.py`.

Edit the `PROMPT` value:

```python
PROMPT = (
    "pixel art, full-body black dragon in side profile facing right, "
    "mouth wide open, actively exhaling bright cyan-blue flames, "
    "isolated game sprite, plain white background"
)
```

Then push the kernel:

```bash
python3 -m kaggle kernels push -p kaggle_sdxl
```

Kaggle kernel:

https://www.kaggle.com/code/gabrielep09/spritelab-sdxl-benchmark

The full 24-image evaluation is in `kaggle_sdxl/benchmark.py`. To run it, change `code_file` in `kaggle_sdxl/kernel-metadata.json` to `benchmark.py` before pushing.

## Local prototype

The local FastAPI interface is a working prototype, but it still uses the legacy SD 1.5 LoRA by default.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start the application:

```bash
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Project structure

```text
app.py                         Local FastAPI service
web/                           Local browser interface
scripts/generate_sprite.py     Local SD 1.5 generator
scripts/01_scrape_sprites.py   Sprite sheet scraper
scripts/02_process_sprites.py  Dataset cleaner and balancer
scripts/sprite_rules.py        Shared dataset filtering rules
kaggle_sdxl/                   Selected SDXL Kaggle workflow
kaggle_flux/                   Rejected FLUX benchmark
kaggle_notebook/               Legacy SD 1.5 LoRA training
```

## Benchmark findings

### SDXL with Pixel Art XL

- Selected model path
- Strong category recognition
- Clean isolated sprites
- Eight step LCM generation peaked at 5.81 GB VRAM
- Full 30 step generation peaked at 5.44 GB VRAM
- Complex action placement can still be inconsistent

### FLUX.2 Klein 4B

- Fit within Kaggle T4 memory
- Tested in FP16 and BF16
- Both Kaggle runs decoded as black images
- Rejected for this project

### Custom SD 1.5 LoRA

- Trained successfully
- Learned a rough pixel art texture
- Weak composition and prompt adherence
- Retained only as a technical baseline

## Data and licensing

Generated datasets, model weights, Kaggle outputs, and scraped game assets are excluded from Git. The scraped GBA assets are copyrighted and must not be represented as CC0 or used as a commercial training dataset.

The selected SDXL workflow uses these external models:

- https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- https://huggingface.co/nerijs/pixel-art-xl
- https://huggingface.co/latent-consistency/lcm-lora-sdxl
