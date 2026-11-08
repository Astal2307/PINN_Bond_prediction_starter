"""Шаг 1. Подготовка данных: чтение папки рынка (форматы ISS Московской биржи и таблица RUONIA ЦБ).

Market(path):
  target, eval_from, eval_till, eval_days       целевая бумага и окно оценки (task.json)
  bonds[secid]  -> BondData                      описание, купоны, колл, история торгов
  ruonia                                         pd.Series, доли (0.16 = 16%), по датам
  zparams                                        параметры G-кривой ЦБ по дням (B1..B3, T1, G1..G9)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

G_COLS = ["B1", "B2", "B3", "T1"] + [f"G{i}" for i in range(1, 10)]


def _block(d, name):
    b = d.get(name)
    return pd.DataFrame(b["data"], columns=b["columns"]) if b else pd.DataFrame()


@dataclass
class Schedule:
    """Купонный график на оси t (годы ACT/365 от origin); суммы в % номинала."""
    coupon_times: np.ndarray       # даты выплат
    coupon_amounts: np.ndarray     # купон за период, % номинала
    period_starts: np.ndarray      # начала купонных периодов
    t_call: float                  # первый колл (np.inf, если нет)
    T: float                       # погашение
    F: float = 100.0
    K: float = 100.0               # цена колла, % номинала

    def _idx(self, t):
        return np.clip(np.searchsorted(self.coupon_times, t, side="right"), 0, len(self.coupon_times) - 1)

    def accrued(self, t):
        t = np.asarray(t, float)
        i = self._idx(t)
        a = self.coupon_amounts[i] * (t - self.period_starts[i]) / (self.coupon_times[i] - self.period_starts[i])
        return np.where((t < self.period_starts[0]) | (t >= self.coupon_times[-1]), 0.0, np.maximum(a, 0.0))

    def coupon_rate(self, t):
        i = self._idx(np.asarray(t, float))
        return self.coupon_amounts[i] / (self.coupon_times[i] - self.period_starts[i])

    def future(self) -> "Schedule":
        """Только потоки после t = 0."""
        m = self.coupon_times > 0
        return Schedule(self.coupon_times[m], self.coupon_amounts[m], self.period_starts[m], self.t_call, self.T,
                        self.F, self.K)


class BondData:
    def __init__(self, folder: Path, secid: str):
        de = _block(json.loads((folder / "securities" / f"{secid}.json").read_text(encoding="utf-8")), "description")
        self.desc = dict(zip(de.name, de.value))
        bz = json.loads((folder / "bondization" / f"{secid}.json").read_text(encoding="utf-8"))
        self.coupons = _block(bz, "coupons")
        self.offers = _block(bz, "offers")
        h = _block(json.loads((folder / "history" / f"{secid}.json").read_text(encoding="utf-8")), "history")
        h["TRADEDATE"] = pd.to_datetime(h.TRADEDATE)
        self.hist = h.sort_values("TRADEDATE").reset_index(drop=True)
        self.secid = secid
        self.face = float(self.desc.get("FACEVALUE") or 1000)

    @property
    def call_dates(self):
        o = self.offers
        if o.empty:
            return []
        return sorted(pd.to_datetime(o[o.offertype == "Колл-опцион"].offerdate).tolist())

    def schedule(self, origin: pd.Timestamp) -> Schedule:
        cd = pd.to_datetime(self.coupons.coupondate)
        st = pd.to_datetime(self.coupons.startdate)
        ct = (cd - origin).dt.days.values / 365.0
        amt = self.coupons.value.values.astype(float) / self.face * 100.0
        calls = self.call_dates
        t_call = (calls[0] - origin).days / 365.0 if calls else np.inf
        return Schedule(ct, amt, (st - origin).dt.days.values / 365.0, t_call, float(ct[-1]))

    def accint_pct(self, days) -> np.ndarray:
        h = self.hist.set_index("TRADEDATE")
        return h.ACCINT.reindex(pd.DatetimeIndex(days)).values.astype(float) / self.face * 100.0


class Market:
    def __init__(self, path):
        self.path = Path(path)
        self.code = self.path.name
        task = json.loads((self.path / "task.json").read_text(encoding="utf-8"))
        self.target = task["target"]
        self.eval_from, self.eval_till = pd.Timestamp(task["eval_from"]), pd.Timestamp(task["eval_till"])
        r = pd.read_csv(self.path / "ruonia.csv", sep=";", decimal=",")
        self.ruonia = pd.Series(r.iloc[:, 1].values.astype(float) / 100.0,
                                index=pd.to_datetime(r.iloc[:, 0], dayfirst=True)).sort_index()
        z = json.loads((self.path / "zcyc.json").read_text(encoding="utf-8"))
        p = _block(z, "params")
        p["tradedate"] = pd.to_datetime(p.tradedate)
        self.zparams = p.set_index("tradedate").sort_index()[G_COLS].astype(float)
        self.bonds = {f.stem: BondData(self.path, f.stem) for f in sorted((self.path / "securities").glob("*.json"))}
        th = self.bonds[self.target].hist
        m = (th.TRADEDATE >= self.eval_from) & (th.TRADEDATE <= self.eval_till)
        self.eval_days = pd.DatetimeIndex(th.TRADEDATE[m].values)

    @property
    def tb(self) -> BondData:
        return self.bonds[self.target]

    def t_of(self, days):
        """Ось времени бейзлайна: годы от eval_from (= time_origin в manifest.json)."""
        return (pd.DatetimeIndex(days) - self.eval_from).days.values / 365.0

    def ruonia_on(self, days):
        s = self.ruonia
        return s.reindex(s.index.union(pd.DatetimeIndex(days))).ffill().bfill().reindex(pd.DatetimeIndex(days)).values
