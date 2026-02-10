# This is needed just because we don't have python >= 3.12
from itertools import islice
from typing import Generator


# Basically copied from itertools docs
def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


def chunk_dict(data: dict, size: int) -> Generator[dict, None, None]:
    """Yields chunks of the dictionary with a maximum size."""
    keys = list(data.keys())
    for i in range(0, len(keys), size):
        yield {k: data[k] for k in keys[i:i + size]}
