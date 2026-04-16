"""
Sequence Transformer — converts 999-form API sequences to user-facing format.

The Claims API stores sequences inverted:  internal = 1000 - user_sequence
    User says "001" → API stores 999
    User says "002" → API stores 998
    User says "010" → API stores 990

This module walks the combined (Details + List) API response and converts
every allowlisted sequence field from 999-form back to user-facing.

KEY RULE:  Only values >= 900 are converted.
  - The formula 1000-x is its own inverse (1000-999=1, 1000-1=999).
  - If we blindly convert ALL values, a field already showing 1 becomes 999.
  - Values >= 900 are clearly 999-form internal values (seq 001–100).
  - Values < 900 are already user-facing and must be left alone.
  - Real-world claims rarely exceed 100 sequences, so 900 is a safe threshold.
"""

from typing import Any, Dict
from copy import deepcopy
from core.logger import get_logger

logger = get_logger(__name__)

# ─── Only convert values at or above this threshold ──────────────────
# 999-form for seq 001 = 999, seq 002 = 998, ..., seq 100 = 900.
# Anything below 900 is already user-facing. Leave it alone.
_THRESHOLD: int = 900

# ─── Allowlisted field names that can hold a 999-form sequence ───────
SEQUENCE_VALUE_KEYS: frozenset = frozenset({
    "sequence",
    "sequenceNumber",
    "claimSequence",
    "claimSeq",
    "tcdClaimSeq",
    "xdtClaimSeqNbr",
    "tcdClaimSeq999sComp",
    "asgSequenceNumber",
    "firstClaimSequence",
    "secondClaimSequence",
    "clmSeqNbr",
    "seqNbr",
    "claimSequenceNumber",
    "claimSequenceNbr",
    "nonMcoClaimSequenceNumber",
    "secondaryClaimSequence",
    "primaryClaimSequence",
    "stdClaimSeq",
    "std2ClaimSeq",
})

# ─── Authorization fields: claimNumber + 3-digit sequence suffix ─────
AUTHORIZATION_CONCAT_KEYS: frozenset = frozenset({
    "authorizationNumber3pr",
    "authorizationNumber",
    "responseAuthorizationNumber",
})

_MAX_DEPTH: int = 64


# =====================================================================
# Helpers
# =====================================================================

def _to_int(value: Any) -> int | None:
    """Extract integer from int, float, or numeric string. None if not possible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value == int(value):
            return int(value)
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except (ValueError, TypeError):
            return None
    return None


def _needs_conversion(n: int) -> bool:
    """True if this value is a 999-form internal sequence (>= 900)."""
    return _THRESHOLD <= n <= 999


def _convert(n: int) -> int:
    """user_facing = 1000 - internal."""
    return 1000 - n


def _replace(original: Any, user_facing: int) -> Any:
    """Return user_facing in the same type as original."""
    if isinstance(original, float):
        return float(user_facing)
    if isinstance(original, int):
        return user_facing
    if isinstance(original, str):
        width = max(len(original.strip()), 3)
        return str(user_facing).zfill(width)
    return user_facing


# =====================================================================
# Recursive walker
# =====================================================================

def _walk(obj: Any, path: str, log: list, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        return obj

    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            p = f"{path}.{key}" if path else key

            # ── Sequence fields ──
            if key in SEQUENCE_VALUE_KEYS:
                n = _to_int(value)
                if n is not None and _needs_conversion(n):
                    uf = _convert(n)
                    nv = _replace(value, uf)
                    if nv != value:
                        log.append({"path": p, "old": value, "new": nv})
                    out[key] = nv
                else:
                    out[key] = value  # already valid or null — skip
                continue

            # ── Auth concat fields (suffix is 3-digit seq) ──
            if key in AUTHORIZATION_CONCAT_KEYS and isinstance(value, str) and len(value) > 3:
                try:
                    suffix_int = int(value[-3:])
                    if _needs_conversion(suffix_int):
                        new_suffix = str(_convert(suffix_int)).zfill(3)
                        nv = value[:-3] + new_suffix
                        if nv != value:
                            log.append({"path": p, "old": value, "new": nv})
                        out[key] = nv
                        continue
                except (ValueError, TypeError):
                    pass

            # ── Recurse ──
            if isinstance(value, (dict, list)):
                out[key] = _walk(value, p, log, depth + 1)
            else:
                out[key] = value

        return out

    if isinstance(obj, list):
        return [_walk(item, f"{path}[{i}]", log, depth + 1) for i, item in enumerate(obj)]

    return obj


# =====================================================================
# Public API
# =====================================================================

def transform_sequences_to_user_format(
    data: Dict[str, Any],
    user_sequence: str,
    claim_number: str = "",
) -> Dict[str, Any]:
    """
    Convert 999-form sequences (>= 900) to user-facing format.
    Values < 900 are already user-facing and are left untouched.

    Args:
        data:          The enriched claim details dict
        user_sequence: User-facing sequence (e.g. "001") — validation only
        claim_number:  Claim number (unused, kept for compat)

    Returns:
        New dict with converted sequences. Original is never mutated.
    """
    if not isinstance(data, dict):
        return data

    if user_sequence is None or str(user_sequence).strip() == "":
        return data
    try:
        user_int = int(str(user_sequence).strip())
    except (ValueError, TypeError):
        return data
    if not (1 <= user_int <= 999):
        return data

    try:
        log: list = []
        result = _walk(deepcopy(data), "", log, 0)
        if log:
            logger.info(f"Sequence transform: {len(log)} field(s) converted")
            for e in log:
                logger.debug(f"  {e['path']}: {e['old']!r} -> {e['new']!r}")
        else:
            logger.info("Sequence transform: 0 fields needed conversion")
        return result
    except Exception as e:
        logger.error(f"Sequence transform failed: {e}. Returning original data.")
        return data
