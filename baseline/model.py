"""Шаг 2. Модель и уравнение.

Факторы: короткая ставка r (Васичек) и кредитный спред s (постоянный во времени: ds = 0).
  dr = kappa (theta - r) dt + sigma dW
Чистая цена C(t, r, s) = V - A(t), V — грязная цена, A — НКД. Уравнение для C (между купонами):
  C_t + kappa (theta - r) C_r + 1/2 sigma^2 C_rr - (r + s) C + coupon_rate(t) - (r + s) A(t) = 0,
  C(T) = 100 (погашение), на купонных датах C непрерывна (купон выплачивается, НКД обнуляется),
  в дату колла t_c: C(t_c-) = min(C(t_c+), K)  (эмитент гасит по K = 100, если выгодно),
  по r и s граничные условия не ставятся (область берётся с запасом, см. pinn.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Vasicek:
    kappa: float
    theta: float
    sigma: float

    @property
    def stat_sd(self):
        return self.sigma / np.sqrt(2 * self.kappa)

    def zcb(self, r, tau):
        """P(t, t + tau) при ставке r."""
        k, th, s = self.kappa, self.theta, self.sigma
        B = (1 - np.exp(-k * tau)) / k
        lnA = (th - s ** 2 / (2 * k ** 2)) * (B - tau) - s ** 2 * B ** 2 / (4 * k)
        return np.exp(lnA - B * r)


@dataclass
class Model:
    rate: Vasicek
    s_level: float                 # уровень спреда (для области сети)

    def manifest_pde(self) -> dict:
        """Уравнение в формате manifest.json (раздел 3.3 постановки)."""
        f = lambda x: repr(float(x))
        k, th, sg = self.rate.kappa, self.rate.theta, self.rate.sigma
        return {"drift": {"r": f"{f(k)}*({f(th)} - r)"},
                "diffusion": {"r,r": f"{f(sg * sg)}"},
                "discount": "r + s",
                "source": "coupon_rate(t) - (r + s)*accrued(t)"}
