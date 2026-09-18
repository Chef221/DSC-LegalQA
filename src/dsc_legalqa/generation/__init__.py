"""Grounded Legal Answer Generation engine."""

from dsc_legalqa.generation.prompt import (
    SYSTEM_INSTRUCTION_VIETNAMESE,
    build_generation_prompt,
)
from dsc_legalqa.generation.engine import GenerationEngine

__all__ = [
    "SYSTEM_INSTRUCTION_VIETNAMESE",
    "build_generation_prompt",
    "GenerationEngine",
]
