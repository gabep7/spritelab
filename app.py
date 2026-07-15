import asyncio
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scripts.generate_sprite import DEFAULT_LORA, DEFAULT_MODEL, generate_image, load_pipeline, save_outputs

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
ASSETS_DIR = ROOT / "assets"
GENERATED_DIR = ROOT / "generated"
GENERATED_DIR.mkdir(exist_ok=True)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    steps: int = Field(default=40, ge=5, le=50)
    guidance: float = Field(default=7.5, ge=1.0, le=15.0)


class GeneratorRuntime:
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.lock = threading.Lock()

    def generate(self, request):
        with self.lock:
            if self.pipeline is None:
                model = os.environ.get("SPRITELAB_MODEL", DEFAULT_MODEL)
                lora = Path(os.environ.get("SPRITELAB_LORA", DEFAULT_LORA))
                scale = float(os.environ.get("SPRITELAB_LORA_SCALE", "1.0"))
                self.pipeline, self.device = load_pipeline(model, lora, scale)
            image = generate_image(
                self.pipeline,
                request.prompt,
                request.seed,
                request.steps,
                request.guidance,
            )
            output = GENERATED_DIR / "latest.png"
            raw_output = save_outputs(image, output)
            return output, raw_output, self.device


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
    return {"status": "ok", "model_loaded": runtime.pipeline is not None}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    output, raw_output, device = await asyncio.to_thread(runtime.generate, request)
    version = time.time_ns()
    return {
        "image_url": f"/generated/{output.name}?v={version}",
        "raw_url": f"/generated/{raw_output.name}?v={version}",
        "device": device,
        "seed": request.seed,
    }
