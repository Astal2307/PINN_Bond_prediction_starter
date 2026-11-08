"""Шаг 5. Состояние на каждый день окна (наивное обращение).

В дни со сделками спред s подбирается так, чтобы сеть дала цену дня (WAPRICE): C(t, r, s) = WAPRICE
(цена убывает по s, бисекция в пределах области сети). Между днями со сделками — линейная интерполяция,
до первой и после последней сделки — ближайшее значение. Шум сделок не фильтруется: это слабое место бейзлайна.
"""
from __future__ import annotations

import numpy as np

from .pinn import Domain, PriceNet, predict


def invert_days(fn, t, r, price, s_lo, s_hi, iters=60):
    lo, hi = np.full(len(t), s_lo), np.full(len(t), s_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = fn(np.c_[t, r, mid])
        hi = np.where(v < price, mid, hi)      # цена ниже наблюдаемой -> спред меньше
        lo = np.where(v >= price, mid, lo)
    return 0.5 * (lo + hi)


def naive_states(net: PriceNet, dom: Domain, t, r, obs, s_default):
    fn = lambda X: predict(net, X)
    eps = 1e-4 * (dom.s_hi - dom.s_lo)            # чуть внутри области: состояния должны лежать в ней строго
    lo, hi = dom.s_lo + eps, dom.s_hi - eps
    if len(obs) == 0:
        return np.full(len(t), np.clip(s_default, lo, hi)), np.array([])
    d = obs.day_idx.values
    s_obs = invert_days(fn, t[d], r[d], obs.price.values, lo, hi)
    ud = np.unique(d)
    s_day = np.array([s_obs[d == u].mean() for u in ud])
    s = np.interp(t, t[ud], s_day)
    return np.clip(s, lo, hi), s_obs
