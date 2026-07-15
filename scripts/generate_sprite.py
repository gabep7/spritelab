import argparse
import os
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, LCMScheduler, StableDiffusionXLPipeline

from scripts.prompt_templates import NEGATIVE_PROMPT, build_prompt
from scripts.sprite_export import save_sprite_outputs

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_PIXEL_LORA = "nerijs/pixel-art-xl"
DEFAULT_LCM_LORA = "latent-consistency/lcm-lora-sdxl"


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


def generate_image(pipeline, device, description, seed=42, mode="quality", category=None):
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

    prompt = build_prompt(description, category=category)
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


def save_outputs(image, output, size=128):
    return save_sprite_outputs(image, output, size=size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--mode", choices=["fast", "quality"], default="quality")
    parser.add_argument("--category", choices=["character", "creature", "weapon", "item", "building", "vehicle", "effect"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, choices=[64, 96, 128], default=128)
    parser.add_argument("--output", type=Path, default=ROOT / "generated_sprite.png")
    args = parser.parse_args()

    pipeline, device = load_pipeline()
    image = generate_image(
        pipeline,
        device,
        args.prompt,
        seed=args.seed,
        mode=args.mode,
        category=args.category,
    )
    outputs = save_outputs(image, args.output, args.size)
    print(
        f"Saved {outputs['sprite']}, {outputs['preview']}, and {outputs['raw']} on {device}"
    )


if __name__ == "__main__":
    main()
