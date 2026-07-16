"""Run the Spritelab web app on a Kaggle T4 behind a cloudflared quick tunnel.

Push with `python3 -m kaggle kernels push -p kaggle_serve`, then grab the
public URL from the kernel log or /kaggle/working/server_info.json.
The server exits on its own after IDLE_MINUTES without requests.
"""

import json
import re
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

IDLE_MINUTES = 30
MAX_HOURS = 8
PORT = 8000
REPO_URL = "https://github.com/gabep7/spritelab.git"
SECRETS_PATH = Path(__file__).with_name("secrets.local")
NTFY_TOPIC = "spritelab-local"
ACCESS_TOKEN = ""
if SECRETS_PATH.exists():
    for line in SECRETS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "TOKEN":
            ACCESS_TOKEN = value.strip()
        elif key.strip() == "NTFY_TOPIC":
            NTFY_TOPIC = value.strip()
WORK = Path("/kaggle/working")
REPO_DIR = WORK / "spritelab"

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "diffusers==0.37.0",
        "transformers==4.57.1",
        "accelerate==1.10.1",
        "peft==0.17.1",
        "safetensors==0.6.2",
        "torchao>=0.16.0",
        "fastapi>=0.115",
        "uvicorn>=0.34",
    ],
    check=True,
)
subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

CLOUDFLARED = WORK / "cloudflared"
urllib.request.urlretrieve(
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    CLOUDFLARED,
)
CLOUDFLARED.chmod(CLOUDFLARED.stat().st_mode | stat.S_IEXEC)

import os
import threading
import time

import peft.import_utils as _peft_import_utils
import peft.tuners.lora.torchao as _peft_torchao

_peft_import_utils.is_torchao_available = lambda: False
_peft_torchao.is_torchao_available = lambda: False

sys.path.insert(0, str(REPO_DIR))
os.environ.setdefault("SPRITELAB_WARMUP", "1")
if ACCESS_TOKEN:
    os.environ["SPRITELAB_TOKEN"] = ACCESS_TOKEN
else:
    raise SystemExit("Missing TOKEN in secrets.local. Refusing to start an open server.")

import uvicorn
from app import app, runtime

started = time.time()
last_activity = time.time()


@app.middleware("http")
async def _track_activity(request, call_next):
    global last_activity
    last_activity = time.time()
    return await call_next(request)


def _watchdog():
    while True:
        time.sleep(60)
        if time.time() - started > MAX_HOURS * 3600:
            print("Max session time reached, shutting down.", flush=True)
            _notify("stopped: max session time")
            os._exit(0)
        busy = runtime.snapshot()["phase"] != "idle"
        if not busy and time.time() - last_activity > IDLE_MINUTES * 60:
            print(f"No requests for {IDLE_MINUTES} minutes, shutting down.", flush=True)
            _notify("stopped: idle timeout")
            os._exit(0)


def _notify(message):
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=message.encode(),
                headers={"Title": "spritelab server"},
            ),
            timeout=10,
        )
    except OSError as error:
        print(f"ntfy publish failed: {error}", flush=True)


def _tunnel():
    process = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    for line in process.stdout:
        match = pattern.search(line)
        if match:
            url = match.group(0)
            banner = "=" * 60
            print(banner, flush=True)
            print(f"SPRITELAB URL: {url}", flush=True)
            print(banner, flush=True)
            (WORK / "server_info.json").write_text(json.dumps({"url": url}))
            _notify(url)
            break
    for _ in process.stdout:
        pass


threading.Thread(target=_watchdog, daemon=True).start()
threading.Thread(target=_tunnel, daemon=True).start()

uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
