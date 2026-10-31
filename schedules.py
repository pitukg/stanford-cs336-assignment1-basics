import math


def cosine_schedule(
    t: int,
    lr_max: float,
    lr_min: float,
    warmup: int,
    final: int
) -> float:
    if t < warmup:
        return t / warmup * lr_max
    elif warmup <= t <= final:
        return lr_min + 1/2 * (1 + math.cos(
            (t - warmup) / (final - warmup) * math.pi
        )) * (lr_max - lr_min)
    else:
        return lr_min
