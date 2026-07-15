NEGATIVE_PROMPT = (
    "3d render, realistic, photograph, blurry, smooth shading, text, words, letters, "
    "watermark, signature, scenery, complex background, cropped, multiple objects, "
    "panel, sheet, collage"
)

PROMPT_PREFIX = "pixel art"
PROMPT_SUFFIX = "full object visible, isolated game sprite, plain white background"

CATEGORY_HINTS = {
    "character": "full-body character, readable silhouette",
    "creature": "full-body creature, readable silhouette",
    "weapon": "single weapon item, centered, no hands",
    "item": "single game item, centered",
    "building": "full structure, readable silhouette",
    "vehicle": "full vehicle, readable silhouette",
    "effect": "single magic or attack effect, clear shape",
}


def build_prompt(description, category=None):
    description = description.strip().rstrip(",")
    parts = [PROMPT_PREFIX]
    if category:
        hint = CATEGORY_HINTS.get(category)
        if hint:
            parts.append(hint)
    parts.append(description)
    parts.append(PROMPT_SUFFIX)
    return ", ".join(parts)


# Presets used by the web UI and the multi-prompt Kaggle generator.
PRESETS = [
    {
        "id": "fire-mage",
        "label": "FIRE MAGE",
        "category": "character",
        "description": "full-body fire mage holding a staff with an orange flame tip",
    },
    {
        "id": "knight",
        "label": "KNIGHT",
        "category": "character",
        "description": "full-body armored knight holding a sword and heater shield",
    },
    {
        "id": "chest",
        "label": "CHEST",
        "category": "item",
        "description": "wooden treasure chest with gold trim, lid slightly open",
    },
    {
        "id": "dog",
        "label": "DOG",
        "category": "creature",
        "description": "small brown dog standing, side view facing right",
    },
    {
        "id": "dragon",
        "label": "DRAGON",
        "category": "creature",
        "description": (
            "full-body black dragon in side profile facing right, mouth open, "
            "breathing a long visible stream of bright cyan-blue fire to the right"
        ),
    },
    {
        "id": "slime",
        "label": "SLIME",
        "category": "creature",
        "description": "cute translucent blue slime with a tiny gold crown",
    },
    {
        "id": "potion",
        "label": "POTION",
        "category": "item",
        "description": "glass potion bottle filled with glowing cyan liquid",
    },
    {
        "id": "airship",
        "label": "AIRSHIP",
        "category": "vehicle",
        "description": "fantasy wooden airship with blue sails and brass fittings",
    },
]
