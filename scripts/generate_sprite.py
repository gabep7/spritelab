import argparse
import os
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, LCMScheduler, StableDiffusionXLPipeline
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_PIXEL_LORA = "nerijs/pixel-art-xl"
DEFAULT_LCM_LORA = "latent-consistency/lcm-lora-sdxl"
NEGATIVE_PROMPT = (
    "3d render, realistic, photograph, blurry, smooth shading, text, words, letters, "
    "watermark, signature, scenery, complex background, cropped, multiple objects"
)


def runtime_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_pipeline(
    model=DEFAULT_MODEL,
    pixel_lora=DEFAULT_PIXEL_LORA,
    lcm_lora=DEFAULT_LCM_LORA,
):
    device, dtype = runtime_device()
    options = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if dtype == torch.float16:
        options["variant"] = "fp16"
    pipeline = StableDiffusionXLPipeline.from_pretrained(model, **options)
    scheduler_config = pipeline.scheduler.config
    pipeline.quality_scheduler = DPMSolverMultistepScheduler.from_config(
        scheduler_config,
        use_karras_sigmas=True,
    )
    pipeline.fast_scheduler = LCMScheduler.from_config(scheduler_config)
    pipeline.load_lora_weights(
        pixel_lora,
        weight_name="pixel-art-xl.safetensors",
        adapter_name="pixel",
    )
    pipeline.load_lora_weights(lcm_lora, adapter_name="lcm")

    use_cpu_offload = os.environ.get("SPRITELAB_CPU_OFFLOAD", "0") == "1"
    if device == "cuda" and use_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    if device == "mps":
        pipeline.enable_attention_slicing()
    return pipeline, device


def generate_image(pipeline, device, description, seed=42, mode="quality"):
    if mode == "fast":
        pipeline.scheduler = pipeline.fast_scheduler
        pipeline.set_adapters(["lcm", "pixel"], adapter_weights=[1.0, 1.2])
        steps = 8
        guidance = 1.5
    elif mode == "quality":
        pipeline.scheduler = pipeline.quality_scheduler
        pipeline.set_adapters("pixel", adapter_weights=1.2)
        steps = 30
        guidance = 7.0
    else:
        raise ValueError(f"Unknown generation mode: {mode}")

    prompt = (
        f"pixel art, {description}, full object visible, isolated game sprite, "
        "plain white background"
    )
    generator_device = "cuda" if device == "cuda" else "cpu"
    return pipeline(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=1024,
        width=1024,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device=generator_device).manual_seed(seed),
    ).images[0]


def extract_sprite(image, size=128):
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
            ImageDraw.floodfill(background, corner, marker, thresh=40)

    rgba = source.convert("RGBA")
    alpha = Image.new("L", source.size, 255)
    alpha.putdata([0 if pixel == marker else 255 for pixel in background.getdata()])
    rgba.putalpha(alpha)
    content_box = alpha.getbbox()
    if content_box is None:
        raise RuntimeError("Generated image contains no foreground")

    content = rgba.crop(content_box)
    content_limit = round(size * 0.82)
    scale = min(content_limit / content.width, content_limit / content.height)
    resized = content.resize(
        (max(1, round(content.width * scale)), max(1, round(content.height * scale))),
        Image.Resampling.NEAREST,
    )
    sprite = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    sprite.alpha_composite(
        resized,
        ((size - resized.width) // 2, (size - resized.height) // 2),
    )
    return sprite


def save_outputs(image, output, size=128):
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(f"{output.stem}_raw.png")
    preview_output = output.with_name(f"{output.stem}_preview.png")
    image.save(raw_output)
    sprite = extract_sprite(image, size)
    sprite.save(output)
    sprite.resize((512, 512), Image.Resampling.NEAREST).save(preview_output)
    return {
        "sprite": output,
        "preview": preview_output,
        "raw": raw_output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--mode", choices=["fast", "quality"], default="quality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, choices=[64, 96, 128], default=128)
    parser.add_argument("--output", type=Path, default=ROOT / "generated_sprite.png")
    args = parser.parse_args()

    pipeline, device = load_pipeline()
    image = generate_image(pipeline, device, args.prompt, args.seed, args.mode)
    outputs = save_outputs(image, args.output, args.size)
    print(
        f"Saved {outputs['sprite']}, {outputs['preview']}, and {outputs['raw']} on {device}"
    )


if __name__ == "__main__":
    main()
