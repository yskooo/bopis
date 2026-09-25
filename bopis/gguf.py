"""Find the ladder's GGUF files on disk, and measure them correctly.

Two jobs, both about not making the operator type ten ``--model`` flags:

1. :func:`discover` scans a directory for the official Qwen2.5-Instruct GGUF
   file names (``qwen2.5-3b-instruct-q4_k_m.gguf`` and so on) and maps each to
   its ``MODEL:VARIANT`` key, so ``--models-dir models`` replaces a flag per
   file.
2. :func:`total_size` measures a model's real footprint. The larger official
   files are split into shards (``...-00001-of-00002.gguf``); llama.cpp loads
   the whole set from the first shard, so the memory guard must count every
   shard, not only the one whose path is passed.

Standard library only.
"""

from __future__ import annotations

import os
import re
from typing import Dict

from bopis import config_space as cs

#: File-name precision tokens as published, mapped to BOPIS's variant names.
_VARIANTS = {"q4_k_m": "Q4_K_M", "q8_0": "Q8_0", "fp16": "F16", "f16": "F16"}

_NAME = re.compile(
    r"^qwen2\.5-(?P<size>\d+(?:\.\d+)?b)-instruct-(?P<variant>q4_k_m|q8_0|fp16|f16)"
    r"(?:-(?P<part>\d{5})-of-(?P<parts>\d{5}))?\.gguf$",
    re.IGNORECASE,
)

_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def parse_name(filename: str):
    """``(model_key, variant, part, parts)`` for an official ladder file, or None."""
    match = _NAME.match(os.path.basename(filename))
    if not match:
        return None
    model_key = f"qwen2.5-{match.group('size').lower()}"
    if model_key not in cs.MODELS:
        return None
    part = int(match.group("part") or 1)
    parts = int(match.group("parts") or 1)
    return model_key, _VARIANTS[match.group("variant").lower()], part, parts


def total_size(path: str) -> int:
    """Bytes of *path*, plus every sibling shard when it is a split GGUF."""
    match = _SHARD.search(path)
    if not match:
        return os.path.getsize(path)
    parts = int(match.group(2))
    total = 0
    for index in range(1, parts + 1):
        shard = _SHARD.sub(f"-{index:05d}-of-{parts:05d}.gguf", path)
        if not os.path.exists(shard):
            raise FileNotFoundError(
                f"{os.path.basename(path)} is split into {parts} parts but "
                f"{os.path.basename(shard)} is missing; the download is incomplete"
            )
        total += os.path.getsize(shard)
    return total


def discover(directory: str) -> Dict[str, str]:
    """``{"MODEL:VARIANT": path_to_first_shard}`` for every complete ladder file.

    Incomplete shard sets are skipped, because llama.cpp would fail to load
    them mid-study.
    """
    found: Dict[str, str] = {}
    if not os.path.isdir(directory):
        return found
    for name in sorted(os.listdir(directory)):
        parsed = parse_name(name)
        if parsed is None:
            continue
        model_key, variant, part, _parts = parsed
        if part != 1:
            continue
        path = os.path.join(directory, name)
        try:
            total_size(path)
        except FileNotFoundError:
            continue
        found[f"{model_key}:{variant}"] = path
    return found


__all__ = ["discover", "parse_name", "total_size"]
