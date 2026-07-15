import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scripts.generate_sprite import (
    DEFAULT_LCM_LORA,
    DEFAULT_MODEL,
    DEFAULT_PIXEL_LORA,
    generate_image,
    load_pipeline,
    save_outputs,
)
from scripts.prompt_templates import PRESETS

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
ASSETS_DIR = ROOT / "assets"
GENERATED_DIR = ROOT / "generated"
GENERATED_DIR.mkdir(exist_ok=True)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    mode: Literal["fast", "quality"] = "quality"
    size: Literal[64, 96, 128] = 128
    category: Optional[
        Literal["character", "creature", "weapon", "item", "building", "vehicle", "effect"]
    ] = None


class GeneratorRuntime:
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.lock = threading.Lock()

    def generate(self, request):
        with self.lock:
            if self.pipeline is None:
                self.pipeline, self.device = load_pipeline(
                    os.environ.get("SPRITELAB_MODEL", DEFAULT_MODEL),
                    os.environ.get("SPRITELAB_PIXEL_LORA", DEFAULT_PIXEL_LORA),
                    os.environ.get("SPRITELAB_LCM_LORA", DEFAULT_LCM_LORA),
                )
            image = generate_image(
                self.pipeline,
                self.device,
                request.prompt,
                request.seed,
                request.mode,
                category=request.category,
            )
            outputs = save_outputs(
                image,
                GENERATED_DIR / "latest.png",
                request.size,
            )
            return outputs, self.device


runtime = GeneratorRuntime()
app = FastAPI(title="Spritelab")
app.mount("/static", StaticFiles(directory=WEB_DIR, check_dir=False), name="static")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR, check_dir=False), name="assets")
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "SDXL + Pixel Art XL",
        "model_loaded": runtime.pipeline is not None,
    }


@app.get("/api/presets")
def presets():
    return {"presets": PRESETS}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    try:
        outputs, device = await asyncio.to_thread(runtime.generate, request)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    version = time.time_ns()
    return {
        "image_url": f"/generated/{outputs['preview'].name}?v={version}",
        "sprite_url": f"/generated/{outputs['sprite'].name}?v={version}",
        "raw_url": f"/generated/{outputs['raw'].name}?v={version}",
        "device": device,
        "mode": request.mode,
        "seed": request.seed,
        "size": request.size,
        "category": request.category,
    }
