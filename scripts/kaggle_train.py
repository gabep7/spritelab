"""
Kaggle notebook for SpriteLab LoRA training.

This script is pushed as a Kaggle notebook and run on a free P100 GPU.
It downloads the GBA sprite dataset, trains a LoRA on SD 1.5, and
saves the output for download.
"""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Offline Kaggle: skip pip; patch peft torchao gate (see spritelab_train.ipynb)
import peft.import_utils as _peft_iu
_peft_iu.is_torchao_available = lambda: False

# === DATASET ===
# Kaggle mounts datasets at /kaggle/input/<dataset-name>/
DATASET_DIR = Path('/kaggle/input/gba-sprites/sprites')
if not DATASET_DIR.exists():
    # Try finding it
    for p in Path('/kaggle/input').rglob('sprites'):
        if p.is_dir():
            DATASET_DIR = p
            break

png_count = len(list(DATASET_DIR.glob('*.png')))
print(f'Dataset: {DATASET_DIR}')
print(f'Images: {png_count} PNG files')
assert png_count > 0, 'No PNG files found!'

# === MODEL ===
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig

MODEL_ID = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
).to('cuda')

vae = pipe.vae
unet = pipe.unet
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder='scheduler')

for m in [unet, vae, text_encoder]:
    m.requires_grad_(False)

unet.add_adapter(LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['to_q', 'to_k', 'to_v', 'to_out.0', 'proj_in', 'proj_out'],
    lora_dropout=0.05,
    bias='none',
))

trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
total = sum(p.numel() for p in unet.parameters())
print(f'Base model loaded.')
print(f'Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

# === DATALOADER ===
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

RESOLUTION = 512

class SpriteDataset(Dataset):
    def __init__(self, dataset_dir, resolution=512):
        self.metadata = []
        meta_path = dataset_dir / 'metadata.jsonl'
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    self.metadata.append(json.loads(line))
        else:
            for img_path in sorted(dataset_dir.glob('*.png')):
                self.metadata.append({
                    'file_name': img_path.name,
                    'caption': 'pixel art sprite, game character, gba style'
                })
        self.dataset_dir = dataset_dir
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        image = Image.open(self.dataset_dir / entry['file_name']).convert('RGB')
        image = self.transform(image)
        caption = entry.get('caption', 'pixel art sprite, game character')
        return {'image': image, 'caption': caption}

dataset = SpriteDataset(DATASET_DIR, RESOLUTION)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
print(f'Dataset size: {len(dataset)} sprites')

# === TRAIN ===
import torch.nn.functional as F
from tqdm.auto import tqdm

LEARNING_RATE = 1e-4
MAX_TRAIN_STEPS = 2000
SAVE_EVERY = 500
GRADIENT_ACCUMULATION = 4

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet.parameters()),
    lr=LEARNING_RATE,
)
unet.train()

step = 0
pbar = tqdm(total=MAX_TRAIN_STEPS, desc='Training')
losses = []

while step < MAX_TRAIN_STEPS:
    for batch in dataloader:
        if step >= MAX_TRAIN_STEPS:
            break

        images = batch['image'].to('cuda', dtype=torch.float16)
        captions = batch['caption']

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample().to(dtype=torch.float16)
            latents = latents * vae.config.scaling_factor

        with torch.no_grad():
            text_inputs = tokenizer(
                captions, padding='max_length', max_length=77,
                truncation=True, return_tensors='pt'
            ).to('cuda')
            encoder_hidden_states = text_encoder(text_inputs['input_ids'])[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device='cuda'
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(noise_pred, noise)
        loss = loss / GRADIENT_ACCUMULATION
        loss.backward()

        if (step + 1) % GRADIENT_ACCUMULATION == 0:
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item() * GRADIENT_ACCUMULATION)
        pbar.set_postfix(loss=f'{loss.item() * GRADIENT_ACCUMULATION:.4f}')
        pbar.update(1)
        step += 1

        if step % SAVE_EVERY == 0:
            avg_loss = sum(losses[-SAVE_EVERY:]) / SAVE_EVERY
            print(f'\nStep {step}: avg loss {avg_loss:.4f}')

pbar.close()
print(f'\nTraining complete. Final avg loss: {sum(losses[-100:]) / min(100, len(losses)):.4f}')

# === SAVE ===
SAVE_DIR = '/kaggle/working/spritelab_lora'
unet.save_pretrained(SAVE_DIR)
print(f'LoRA adapter saved to {SAVE_DIR}')

# === GENERATE TEST SPRITES ===
from PIL import Image as PILImage
import numpy as np

def pixelate(img, target_size=64):
    small = img.resize((target_size, target_size), PILImage.NEAREST)
    return small.resize(img.size, PILImage.NEAREST)

def quantize_palette(img, colors=32):
    return img.convert('P', palette=PILImage.ADAPTIVE, colors=colors).convert('RGB')

prompts = [
    'fire mage character, gba style, white background',
    'knight with sword, gba style, white background',
    'blue dragon, gba style, white background',
    'female archer, gba style, white background',
]

for i, p in enumerate(prompts):
    img = pipe(p, num_inference_steps=30, guidance_scale=7.5, width=512, height=512).images[0]
    img = pixelate(img, target_size=64)
    img = quantize_palette(img, colors=32)
    img.save(f'/kaggle/working/sprite_{i:02d}.png')
    print(f'saved sprite_{i:02d}.png: {p}')

print('\nDone! Download spritelab_lora/ and sprite_*.png from the Output tab.')