"""Сравнение долей двух групп: z-тест, 95% доверительный интервал разницы, знаковый тест."""

import math
from statistics import NormalDist

import pandas as pd

Z_95 = NormalDist().inv_cdf(0.975)


def z_test(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float]:
    pooled = (k1 + k0) / (n1 + n0)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    z = (k1 / n1 - k0 / n0) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


# Для z-статистики берётся объединённая доля (гипотеза о равенстве долей), для интервала —
# раздельные доли групп (интервал Вальда для разности).
def wald_interval(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float]:
    p1, p0 = k1 / n1, k0 / n0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return p1 - p0 - Z_95 * se, p1 - p0 + Z_95 * se


def compare_shares(target: pd.Series, flag: pd.Series) -> dict[str, float]:
    k1, n1 = int(target[flag].sum()), int(flag.sum())
    k0, n0 = int(target[~flag].sum()), int((~flag).sum())
    z, p_value = z_test(k1, n1, k0, n0)
    low, high = wald_interval(k1, n1, k0, n0)
    return {
        "n1": n1,
        "share1": k1 / n1,
        "n0": n0,
        "share0": k0 / n0,
        "diff": k1 / n1 - k0 / n0,
        "ci_low": low,
        "ci_high": high,
        "z": z,
        "p_value": p_value,
    }


def sign_test(negative: int, total: int) -> float:
    tail = sum(math.comb(total, i) for i in range(min(negative, total - negative) + 1))
    return min(1.0, 2 * tail / 2**total)


def channel_differences(data: pd.DataFrame, target: str, flag_column: str) -> pd.Series:
    differences = {}
    for channel, part in data.groupby("channel"):
        flag = part[flag_column].astype(bool)
        if flag.any() and not flag.all():
            differences[channel] = part.loc[flag, target].mean() - part.loc[~flag, target].mean()
    return pd.Series(differences, dtype=float)


def sign_test_row(differences: pd.Series) -> dict[str, float]:
    negative = int((differences < 0).sum())
    return {
        "channels": len(differences),
        "negative": negative,
        "positive": int((differences > 0).sum()),
        "median_diff": differences.median(),
        "p_value": sign_test(negative, len(differences)),
    }
