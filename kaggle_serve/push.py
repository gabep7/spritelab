"""Stage secrets into a temp kernel dir and push. Never commits secrets."""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SECRETS = ROOT / "secrets.local"


def load_secrets():
    values = {}
    if not SECRETS.exists():
        raise SystemExit(
            f"Missing {SECRETS}. Create it with:\n"
            "TOKEN=...long random...\n"
            "NTFY_TOPIC=...private random topic...\n"
        )
    for line in SECRETS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    if "TOKEN" not in values:
        raise SystemExit("secrets.local must define TOKEN=...")
    if "NTFY_TOPIC" not in values:
        raise SystemExit("secrets.local must define NTFY_TOPIC=...")
    return values


def main():
    secrets = load_secrets()
    source = (ROOT / "serve.py").read_text()
    if "___SPRITELAB_TOKEN___" not in source or "___SPRITELAB_NTFY___" not in source:
        raise SystemExit("serve.py is missing secret placeholders")

    staged = source.replace("___SPRITELAB_TOKEN___", secrets["TOKEN"]).replace(
        "___SPRITELAB_NTFY___", secrets["NTFY_TOPIC"]
    )
    if re.search(r"___SPRITELAB_[A-Z]+___", staged):
        raise SystemExit("unreplaced secret placeholders remain")

    with tempfile.TemporaryDirectory(prefix="spritelab-serve-") as tmp:
        staging = Path(tmp)
        (staging / "serve.py").write_text(staged)
        shutil.copy2(ROOT / "kernel-metadata.json", staging / "kernel-metadata.json")
        print("Pushing locked Kaggle server kernel...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(staging)],
            check=True,
        )
        print(
            "Pushed. Open the ntfy topic from secrets.local for the URL, then append ?token=...",
            flush=True,
        )


if __name__ == "__main__":
    main()
