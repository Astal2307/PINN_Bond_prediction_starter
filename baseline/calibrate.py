"""Шаг 3. Калибровка по данным рынка.

* Васичек по RUONIA: средние за неделю, точная AR(1)-схема OU (МНК), kappa ограничена [0.05, 6].
* Состояние r на каждый день окна: RUONIA этого дня (последнее известное значение).
* Спред: постоянный; его уровень и разброс по дням окна — грубая оценка по сделкам целевой бумаги
  (цена дня WAPRICE, спред подбирается так, чтобы цена Васичека без опциона совпала с ценой сделки).
  Эта оценка задаёт только область сети по s; сам спред по дням восстанавливается на шаге 5 по сети.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Market, Schedule
from .model import Vasicek, Model

KMIN, KMAX = 0.05, 6.0


def vasicek_ruonia(mkt: Market) -> Vasicek:
    w = mkt.ruonia[mkt.ruonia.index <= mkt.eval_till].resample("W").mean().dropna().values
    dt = 7 / 365
    b, a = np.polyfit(w[:-1], w[1:], 1)
    b = float(np.clip(b, np.exp(-KMAX * dt), np.exp(-KMIN * dt)))
    a = float(np.mean(w[1:]) - b * np.mean(w[:-1]))
    k = -np.log(b) / dt
    th = a / (1 - b)
    e = w[1:] - a - b * w[:-1]
    sg = float(np.std(e) * np.sqrt(2 * k / (1 - b * b)))
    return Vasicek(float(k), float(th), sg)


def straight_spread(vas: Vasicek, sch: Schedule, t, r, dirty):
    """Спред s, при котором сумма потоков до колла (без опциона) по Васичеку с exp(-s tau) равна dirty.
    Векторная бисекция по дням."""
    t_end = min(sch.t_call, sch.T)
    lo, hi = np.full(len(t), -0.5), np.full(len(t), 1.5)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v = np.zeros(len(t))
        for ti, ci in zip(sch.coupon_times, sch.coupon_amounts):
            if ti > t_end + 1e-9:
                break
            tau = ti - t
            m = tau > 1e-9
            cf = ci + (100.0 if abs(ti - t_end) < 1e-9 else 0.0)
            v = v + np.where(m, cf * vas.zcb(r, np.maximum(tau, 0)) * np.exp(-mid * np.maximum(tau, 0)), 0.0)
        hi = np.where(v < dirty, mid, hi)
        lo = np.where(v >= dirty, mid, lo)
    return 0.5 * (lo + hi)


def traded_days(mkt: Market, days=None) -> pd.DataFrame:
    """Дни со сделками целевой бумаги в окне: дата, индекс дня окна, WAPRICE, объём."""
    h = mkt.tb.hist
    days = mkt.eval_days if days is None else days
    h = h[(h.NUMTRADES > 0) & h.TRADEDATE.isin(days) & h.WAPRICE.notna()]
    return pd.DataFrame(dict(date=h.TRADEDATE.values, day_idx=pd.DatetimeIndex(days).get_indexer(h.TRADEDATE),
                             price=h.WAPRICE.values.astype(float), volume=h.VOLUME.values.astype(float)))


def calibrate(mkt: Market):
    vas = vasicek_ruonia(mkt)
    t = mkt.t_of(mkt.eval_days)
    r = mkt.ruonia_on(mkt.eval_days)
    sch = mkt.tb.schedule(mkt.eval_from)
    obs = traded_days(mkt)
    acc = mkt.tb.accint_pct(mkt.eval_days)
    t_end = min(sch.t_call, sch.T)
    s_obs = np.array([])
    if len(obs):
        d = obs.day_idx.values
        far = t_end - t[d] >= 0.25                          # у самого колла цена почти не зависит от спреда
        if far.any():
            s_obs = straight_spread(vas, sch, t[d][far], r[d][far], obs.price.values[far] + acc[d][far])
    if len(s_obs) == 0:                                     # нет сделок в окне: вся история бумаги
        h = mkt.tb.hist
        h = h[(h.NUMTRADES > 0) & h.WAPRICE.notna() & (h.TRADEDATE < mkt.eval_from)]
        if len(h):
            th = mkt.t_of(h.TRADEDATE)
            ah = mkt.tb.accint_pct(h.TRADEDATE)
            s_obs = straight_spread(vas, sch, th, mkt.ruonia_on(h.TRADEDATE), h.WAPRICE.values + ah)
    level = float(np.median(s_obs)) if len(s_obs) else 0.03
    s_lo = min(level - 0.10, float(np.min(s_obs)) - 0.05 if len(s_obs) else level - 0.10)
    s_hi = max(level + 0.10, float(np.max(s_obs)) + 0.05 if len(s_obs) else level + 0.10)
    info = dict(kappa=vas.kappa, theta=vas.theta, sigma=vas.sigma, s_level=level, s_lo=s_lo, s_hi=s_hi,
                n_trades_window=int(len(obs)))
    return Model(vas, level), dict(t=t, r=r, sch=sch, obs=obs, s_range=(s_lo, s_hi)), info
