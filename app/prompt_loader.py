from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def render_prompt(name: str, replacements: Mapping[str, str] | None = None) -> str:
    text = load_prompt(name)
    for key, value in (replacements or {}).items():
        text = text.replace(f"{{{{{key}}}}}", value)

    unresolved = sorted(set(_PLACEHOLDER_PATTERN.findall(text)))
    if unresolved:
        raise ValueError(f"Unresolved prompt placeholders in {name}: {', '.join(unresolved)}")

    return text
