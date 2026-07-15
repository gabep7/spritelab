import argparse
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_LORA = ROOT / "kaggle_output" / "v17" / "spritelab_lora" / "pytorch_lora_weights.safetensors"


def runtime_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_pipeline(model, lora_path, lora_scale):
    device, dtype = runtime_device()
    model_path = Path(model)
    options = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if model_path.is_file():
        pipeline = StableDiffusionPipeline.from_single_file(model_path, **options)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(model, **options)

    pipeline.unet.add_adapter(
        LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
    )
    state = load_file(str(lora_path))
    if lora_scale != 1.0:
        state = {
            key: value * lora_scale if ".lora_B." in key else value
            for key, value in state.items()
        }
    result = set_peft_model_state_dict(pipeline.unet, state)
    if result.unexpected_keys:
        raise RuntimeError(f"LoRA has {len(result.unexpected_keys)} unexpected keys")
    loaded_state = get_peft_model_state_dict(pipeline.unet)
    if loaded_state.keys() != state.keys() or any(
        not torch.equal(loaded_state[key], value) for key, value in state.items()
    ):
        raise RuntimeError("LoRA tensors did not load exactly")

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.eval()
    pipeline = pipeline.to(device)
    if device == "mps":
        pipeline.enable_attention_slicing()
    return pipeline, device


def postprocess(image, pixel_size, colors):
    pixelated = image.resize((pixel_size, pixel_size), Image.Resampling.NEAREST)
    pixelated = pixelated.resize(image.size, Image.Resampling.NEAREST)
    return pixelated.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors).convert("RGB")


def generate_image(pipeline, description, seed=42, steps=40, guidance=7.5):
    prompt = f"pixel art sprite, {description}, gba style, isolated on white background"
    return pipeline(
        prompt,
        negative_prompt="text, watermark, background scenery, blurry, smooth shading",
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=256,
        height=256,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]


def save_outputs(image, output, pixel_size=64, colors=32):
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(f"{output.stem}_raw{output.suffix}")
    image.save(raw_output)
    postprocess(image, pixel_size, colors).save(output)
    return raw_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--pixel-size", type=int, default=64)
    parser.add_argument("--colors", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "generated_sprite.png")
    args = parser.parse_args()

    if not args.lora.is_file():
        raise FileNotFoundError(args.lora)
    pipeline, device = load_pipeline(args.model, args.lora, args.lora_scale)
    image = generate_image(pipeline, args.prompt, args.seed, args.steps, args.guidance)
    raw_output = save_outputs(image, args.output, args.pixel_size, args.colors)
    print(f"Saved {args.output} and {raw_output} on {device}")


if __name__ == "__main__":
    main()
