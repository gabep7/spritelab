from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prompt_templates import PRESETS, build_prompt


def test_build_prompt_includes_isolation_suffix():
    prompt = build_prompt("blue slime with a crown", category="creature")
    assert prompt.startswith("pixel art")
    assert "blue slime with a crown" in prompt
    assert "isolated game sprite" in prompt
    assert "plain white background" in prompt
    assert "readable silhouette" in prompt


def test_build_prompt_without_category():
    prompt = build_prompt("wooden chest")
    assert "wooden chest" in prompt
    assert "readable silhouette" not in prompt


def test_presets_are_complete():
    assert len(PRESETS) >= 4
    for preset in PRESETS:
        assert preset["id"]
        assert preset["label"]
        assert preset["description"]
        assert preset["category"] in {
            "character",
            "creature",
            "weapon",
            "item",
            "building",
            "vehicle",
            "effect",
        }
