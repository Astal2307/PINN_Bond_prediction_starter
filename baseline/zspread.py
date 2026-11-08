"""Шаг 6. Z-спред по фиксированной конвенции (раздел 3.2 постановки).

dirty = clean + ACCINT / 10;  dirty = sum_k CF_k exp(-(G(tau_k) + z) tau_k),
CF_k — купоны до первого колла включительно и номинал 100 в дату колла (без колла — до погашения),
tau = дни / 365, G — G-кривая ЦБ этого дня по параметрам zcyc.params (непрерывное начисление).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Market

_A = [0.0, 0.6]
for _i in range(2, 9):
    _A.append(_A[-1] + 0.6 * 1.6 ** (_i - 1))
_BW = [_A[1]]
for _i in range(1, 9):
    _BW.append(_BW[-1] * 1.6)


def gcurve(p, tau):
    """G(tau) ЦБ в долях (непрерывное начисление); p = [B1, B2, B3, T1, G1..G9], B* в б.п."""
    tau = np.asarray(tau, float)
    b0, b1, b2, t1 = p[:4]
    g = b0 + (b1 + b2) * t1 / tau * (1 - np.exp(-tau / t1)) - b2 * np.exp(-tau / t1)
    for k in range(9):
        g = g + p[4 + k] * np.exp(-((tau - _A[k]) ** 2) / _BW[k] ** 2)
    return g / 1e4


def zspread_bp(mkt: Market, days, clean) -> np.ndarray:
    tb = mkt.tb
    days = pd.DatetimeIndex(days)
    clean = np.asarray(clean, float)
    acc = tb.accint_pct(days)
    P = mkt.zparams.reindex(mkt.zparams.index.union(days)).ffill().reindex(days).values
    cd = pd.DatetimeIndex(pd.to_datetime(tb.coupons.coupondate))
    amt = tb.coupons.value.values.astype(float) / tb.face * 100.0
    calls = tb.call_dates
    end = calls[0] if calls else cd[-1]
    out = np.full(len(days), np.nan)
    for k, d in enumerate(days):
        m = (cd > d) & (cd <= end)
        if not m.any() or not np.isfinite(clean[k]):
            continue
        tau = (cd[m] - d).days.values / 365.0
        cf = amt[m].copy()
        cf[-1] += 100.0
        if cd[m][-1] != end:
            tau, cf = np.r_[tau, (end - d).days / 365.0], np.r_[cf, 100.0]
        w = cf * np.exp(-gcurve(P[k], tau) * tau)
        target = clean[k] + acc[k]
        z = 0.0
        for _ in range(100):                     # Ньютон: функция убывает и выпукла по z
            e = np.exp(-z * tau)
            f, df = np.sum(w * e) - target, -np.sum(w * tau * e)
            step = f / df
            z -= step
            if abs(step) < 1e-12:
                break
        out[k] = z * 1e4
    return out
