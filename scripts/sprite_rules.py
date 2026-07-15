import html

BLOCKED_SHEET_TERMS = (
    "box art",
    "commanding officer",
    "company logo",
    "credits",
    "cutscene",
    "ending",
    "hud",
    "instruction",
    "intro ",
    "intro and",
    "introduction",
    "level select",
    "logo screen",
    "manual cover",
    "menu",
    "name insert",
    "profile",
    "portrait",
    "start screen",
    "startup screen",
    "title screen",
)
BLOCKED_SHEET_PREFIXES = (
    "battle background",
    "cave ",
    "chambers ",
    "forest ",
    "game font",
    "overworld tileset",
    "palace ",
    "sea of ",
)

BLOCKED_SHEET_SUFFIXES = (
    " background",
    " backgrounds",
    " cabin",
    " castle",
    " cave",
    " city",
    " den",
    " desert",
    " forest",
    " house",
    " interior",
    " island",
    " mountain",
    " rooms",
    " selection",
    " stable",
    " store",
    " tileset",
    " tomb",
    " tower",
    " town",
)

BLOCKED_SHEET_SUBJECTS = {
    "abandoned ship",
    "albrook",
    "baren falls",
    "backgrounds",
    "caves / pirate hideout",
    "desert",
    "indigo plateau",
    "interior",
    "miscellaneous",
    "the blackjack",
}


VEHICLE_TERMS = (
    "air force",
    "apc",
    "artillery",
    "battleship",
    "bomber",
    "cruiser",
    "fighter plane",
    "missile",
    "recon",
    "rocket",
    "tank",
)


def sheet_subject(sheet_name):
    return html.unescape(sheet_name).split(" - ", 1)[0].strip()


def include_sheet(subject):
    normalized = subject.casefold()
    return (
        normalized not in BLOCKED_SHEET_SUBJECTS
        and not normalized.startswith(BLOCKED_SHEET_PREFIXES)
        and not normalized.endswith(BLOCKED_SHEET_SUFFIXES)
        and not any(term in normalized for term in BLOCKED_SHEET_TERMS)
    )


def subject_kind(subject):
    normalized = subject.casefold()
    if any(term in normalized for term in VEHICLE_TERMS):
        return "vehicle"
    if "base" in normalized:
        return "building"
    if any(term in normalized for term in ("lizard", "boss")):
        return "creature"
    return "character"
