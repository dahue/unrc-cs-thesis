"""Deterministic sampling helpers for experiment datasets."""

import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import TypeVar


SAMPLING_SEED = 42
DIFFICULTIES = ("easy", "medium", "hard", "extra")

T = TypeVar("T")


def sample_per_difficulty(
    records: Iterable[T],
    limit: int | None,
    difficulty_getter: Callable[[T], str | None],
) -> list[T]:
    """Return up to ``limit`` deterministic random records per difficulty.

    A missing limit preserves every record and its original order. When a limit
    is provided, records without a recognized Spider difficulty are excluded.
    Each difficulty has an independent random stream so its sample is unchanged
    when other difficulties are added to or removed from the request.
    """
    records = list(records)
    if limit is None:
        return records
    if limit < 0:
        raise ValueError("limit must be greater than or equal to zero")

    grouped: dict[str, list[T]] = defaultdict(list)
    for record in records:
        difficulty = difficulty_getter(record)
        if difficulty in DIFFICULTIES:
            grouped[difficulty].append(record)

    sampled: list[T] = []
    for difficulty in DIFFICULTIES:
        candidates = grouped[difficulty]
        rng = random.Random(f"{SAMPLING_SEED}:{difficulty}")
        sampled.extend(rng.sample(candidates, k=min(limit, len(candidates))))

    return sampled
