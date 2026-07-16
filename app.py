import asyncio
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
HISTORY_LIMIT = 60


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    mode: Literal["fast", "quality"] = "quality"
    size: Literal[64, 96, 128] = 128
    category: Optional[
        Literal["character", "creature", "weapon", "item", "building", "vehicle", "effect"]
    ] = None


class HistoryStore:
    def __init__(self, path, limit=HISTORY_LIMIT):
        self.path = path
        self.limit = limit
        self.lock = threading.Lock()
        try:
            entries = json.loads(path.read_text())
        except (OSError, ValueError):
            entries = []
        self.entries = entries if isinstance(entries, list) else []

    def add(self, entry):
        with self.lock:
            self.entries.append(entry)
            dropped = self.entries[: -self.limit]
            self.entries = self.entries[-self.limit :]
            self._save()
        for old in dropped:
            for suffix in ("", "_preview", "_raw"):
                (GENERATED_DIR / f"{old['id']}{suffix}.png").unlink(missing_ok=True)

    def list(self):
        with self.lock:
            return list(reversed(self.entries))

    def _save(self):
        swap = self.path.with_suffix(".json.tmp")
        swap.write_text(json.dumps(self.entries, indent=2))
        os.replace(swap, self.path)


class GeneratorRuntime:
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.progress = {"phase": "idle", "step": 0, "total": 0}

    def _set_progress(self, phase, step=0, total=0):
        with self.state_lock:
            self.progress = {"phase": phase, "step": step, "total": total}

    def snapshot(self):
        with self.state_lock:
            state = dict(self.progress)
        state["model_loaded"] = self.pipeline is not None
        return state

    def _load_locked(self):
        if self.pipeline is None:
            self._set_progress("loading")
            self.pipeline, self.device = load_pipeline(
                os.environ.get("SPRITELAB_MODEL", DEFAULT_MODEL),
                os.environ.get("SPRITELAB_PIXEL_LORA", DEFAULT_PIXEL_LORA),
                os.environ.get("SPRITELAB_LCM_LORA", DEFAULT_LCM_LORA),
            )

    def warmup(self):
        with self.lock:
            try:
                self._load_locked()
            finally:
                self._set_progress("idle")

    def generate(self, request, output):
        with self.lock:
            try:
                self._load_locked()
                self._set_progress("generating")
                image = generate_image(
                    self.pipeline,
                    self.device,
                    request.prompt,
                    request.seed,
                    request.mode,
                    category=request.category,
                    on_step=lambda step, total: self._set_progress(
                        "generating", step, total
                    ),
                )
                self._set_progress("exporting")
                outputs = save_outputs(image, output, request.size)
                return outputs, self.device
            finally:
                self._set_progress("idle")


runtime = GeneratorRuntime()
history = HistoryStore(GENERATED_DIR / "history.json")


@asynccontextmanager
async def lifespan(_app):
    if os.environ.get("SPRITELAB_WARMUP", "0") == "1":
        threading.Thread(target=runtime.warmup, daemon=True).start()
    yield


app = FastAPI(title="Spritelab", lifespan=lifespan)
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


@app.get("/api/progress")
def progress():
    return runtime.snapshot()


@app.get("/api/history")
def list_history():
    return {"entries": history.list()}


@app.get("/api/presets")
def presets():
    return {"presets": PRESETS}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    sprite_id = f"sprite-{stamp}-{uuid.uuid4().hex[:6]}"
    try:
        outputs, device = await asyncio.to_thread(
            runtime.generate, request, GENERATED_DIR / f"{sprite_id}.png"
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    entry = {
        "id": sprite_id,
        "prompt": request.prompt,
        "seed": request.seed,
        "mode": request.mode,
        "size": request.size,
        "category": request.category,
        "device": device,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_url": f"/generated/{outputs['preview'].name}",
        "sprite_url": f"/generated/{outputs['sprite'].name}",
        "raw_url": f"/generated/{outputs['raw'].name}",
    }
    history.add(entry)
    return entry
