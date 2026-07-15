# Spritelab

<p align="center">
  <img src="assets/logo-lockup.png" alt="Spritelab" width="640">
</p>

Spritelab is an experimental pixel art sprite generator built around Stable Diffusion XL and the Pixel Art XL adapter. The selected path is local GPU generation with transparent sprite export. Kaggle remains available for batch jobs on a Tesla T4.

## Current model path

The selected pipeline is:

- Stable Diffusion XL Base 1.0
- Pixel Art XL
- Optional LCM adapter for eight step generation
- 1024 by 1024 generation
- Transparent crop and nearest neighbor export at 64, 96, or 128

The SDXL workflow produces recognizable dragons, mages, knights, archers, slimes, robots, treasure chests, swords, airships, towers, potions, and dogs. Quality mode peaks around 5.5 GB VRAM on a Kaggle T4.

## Generated showcase

<p align="center">
  <img src="assets/showcase-grid.png" alt="Spritelab generated sprite examples" width="900">
</p>

The logo mark and every image above were generated with the Spritelab SDXL workflow. The selected examples include a dragon, skeleton knight, forest witch, robot, crowned slime, fantasy airship, magic potion, and treasure chest.

The earlier SD 1.5 LoRA remains only as a historical baseline. It is not the selected model.

## Local web app

The FastAPI app runs the selected SDXL pipeline and exports cropped transparent PNGs.

Requirements:

- CUDA GPU with about 8 GB VRAM recommended
- On lower VRAM machines set `SPRITELAB_CPU_OFFLOAD=1`

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start the application:

```bash
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

CLI generation:

```bash
python3 -m scripts.generate_sprite "blue slime with a gold crown" --mode quality --size 128
```

## Kaggle batch generator

The multi-prompt batch script is `kaggle_sdxl/generate.py`.

Edit `JOBS`, `MODE`, `SEEDS`, and `EXPORT_SIZE`, then push:

```bash
python3 -m kaggle kernels push -p kaggle_sdxl
```

Kernel:

https://www.kaggle.com/code/gabrielep09/spritelab-sdxl-benchmark

Pinned Kaggle package versions live in `requirements-kaggle.txt`.

Related scripts:

- `kaggle_sdxl/benchmark.py` full 24 image evaluation
- `kaggle_sdxl/showcase.py` logo and marketing asset generation
- `kaggle_sdxl/sprite_export.py` transparent crop helper used by the batch generator

## Project structure

```text
app.py                         Local FastAPI service
web/                           Local browser interface
scripts/generate_sprite.py     SDXL pipeline
scripts/sprite_export.py       Transparent crop and export
scripts/prompt_templates.py    Shared prompts and presets
kaggle_sdxl/                   Batch and benchmark scripts
assets/                        Logo and showcase images
tests/                         Unit tests for export and prompts
kaggle_notebook/               Legacy SD 1.5 LoRA training
kaggle_flux/                   Rejected FLUX benchmark
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
