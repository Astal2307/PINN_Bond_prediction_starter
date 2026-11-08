"""Шаг 4. PINN: одна полносвязная сеть C(t, r, s) решает уравнение из model.py.

Сеть: вход (t, r, s), нормированный на область в [-1, 1]; MLP 4 x 64, tanh; выход 100 + N (чистая цена, % номинала).
Потери: квадрат невязки уравнения в случайных точках области (перевыбор каждые 50 итераций)
  + условие погашения C(T) = 100 + мягкое условие колла C(t_c-) = min(C(t_c+), K).
Оптимизация: Adam (lr 2e-3, экспоненциальное затухание до 5%), затем L-BFGS на фиксированном наборе точек.
Это намеренно простая сеть (без признака фазы купона, без разбиения по колл-дате): её можно улучшать.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .data import Schedule
from .model import Model


@dataclass
class TrainOpts:
    width: int = 64
    depth: int = 4
    n_colloc: int = 2048
    n_bnd: int = 512
    adam_iters: int = 3000
    lbfgs_iters: int = 100
    lr: float = 2e-3
    resample_every: int = 50
    dom_width: float = 3.0           # полуширина области по r в стационарных с.к.о.
    seed: int = 0


@dataclass
class Domain:
    t0: float
    t1: float
    r_lo: float
    r_hi: float
    s_lo: float
    s_hi: float

    @property
    def lo(self):
        return [self.t0, self.r_lo, self.s_lo]

    @property
    def hi(self):
        return [self.t1, self.r_hi, self.s_hi]


def make_domain(model: Model, sch: Schedule, r_states, s_range, width=3.0) -> Domain:
    v = model.rate
    r_lo, r_hi = v.theta - width * v.stat_sd, v.theta + width * v.stat_sd
    r_lo, r_hi = min(r_lo, float(np.min(r_states)) - 0.01), max(r_hi, float(np.max(r_states)) + 0.01)
    s_lo, s_hi = model.s_level - width * 0.01, model.s_level + width * 0.01
    s_lo, s_hi = min(s_lo, s_range[0]), max(s_hi, s_range[1])
    return Domain(0.0, sch.T, r_lo, r_hi, s_lo, s_hi)


class PriceNet(nn.Module):
    def __init__(self, dom: Domain, width=64, depth=4):
        super().__init__()
        self.register_buffer("lo", torch.tensor(dom.lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(dom.hi, dtype=torch.float32))
        layers, d = [], 3
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = 2 * (x - self.lo) / (self.hi - self.lo) - 1
        return 100.0 + self.mlp(z)


class BondTorch:
    """Купонный график в torch: скорость начисления купона и НКД (для источника уравнения)."""

    def __init__(self, sch: Schedule, device):
        f = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float32, device=device)
        self.ct, self.ps, self.amt = f(sch.coupon_times), f(sch.period_starts), f(sch.coupon_amounts)

    def _i(self, t):
        return torch.clamp((t.unsqueeze(-1) >= self.ct).sum(-1), 0, len(self.ct) - 1)

    def rate(self, t):
        i = self._i(t)
        return self.amt[i] / (self.ct[i] - self.ps[i])

    def accrued(self, t):
        i = self._i(t)
        return self.amt[i] * (t - self.ps[i]) / (self.ct[i] - self.ps[i])


class Trainer:
    def __init__(self, model: Model, sch: Schedule, dom: Domain, opts: TrainOpts, device="cpu"):
        self.m, self.sch, self.dom, self.o, self.dev = model, sch.future(), dom, opts, torch.device(device)
        torch.manual_seed(opts.seed)
        self.rng = np.random.default_rng(opts.seed)
        self.bt = BondTorch(self.sch, self.dev)
        v = model.rate
        self.k, self.th, self.sg = float(v.kappa), float(v.theta), float(v.sigma)
        self.has_call = np.isfinite(self.sch.t_call) and dom.t0 < self.sch.t_call < self.sch.T
        self.log = []

    def tens(self, x, grad=False):
        return torch.tensor(x, dtype=torch.float32, device=self.dev, requires_grad=grad)

    def sample(self, n, t_lo, t_hi):
        d = self.dom
        u = self.rng.random((n, 3))
        return np.c_[t_lo + (t_hi - t_lo) * u[:, 0], d.r_lo + (d.r_hi - d.r_lo) * u[:, 1],
                     d.s_lo + (d.s_hi - d.s_lo) * u[:, 2]]

    def residual(self, net, X):
        C = net(X)
        g = torch.autograd.grad(C.sum(), X, create_graph=True)[0]
        g2 = torch.autograd.grad(g[:, 1].sum(), X, create_graph=True)[0]
        t, r, s = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        Ct, Cr, Crr = g[:, 0:1], g[:, 1:2], g2[:, 1:2]
        tt = t.squeeze(1)
        return (Ct + self.k * (self.th - r) * Cr + 0.5 * self.sg ** 2 * Crr - (r + s) * C
                + self.bt.rate(tt).unsqueeze(1) - (r + s) * self.bt.accrued(tt).unsqueeze(1))

    def losses(self, net, x):
        L = {"res": (self.residual(net, self.tens(x, True)) ** 2).mean()}
        nb, sch = self.o.n_bnd, self.sch
        xb = self.sample(nb, sch.T, sch.T)
        L["term"] = ((net(self.tens(xb)) - sch.F) ** 2).mean()
        if self.has_call:
            xc = self.sample(nb, sch.t_call, sch.t_call)
            xm, xp = xc.copy(), xc.copy()
            xm[:, 0] -= 1e-3
            xp[:, 0] += 1e-3
            L["call"] = ((net(self.tens(xm)) - torch.clamp(net(self.tens(xp)), max=sch.K)) ** 2).mean()
        return L

    def train(self) -> PriceNet:
        o, d = self.o, self.dom
        t_start = time.time()
        net = PriceNet(d, o.width, o.depth).to(self.dev)
        params = list(net.parameters())
        opt = torch.optim.Adam(params, lr=o.lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.05 ** (1 / max(o.adam_iters, 1)))
        x = self.sample(o.n_colloc, d.t0, d.t1)
        for it in range(o.adam_iters):
            if it % o.resample_every == 0:
                x = self.sample(o.n_colloc, d.t0, d.t1)
            L = self.losses(net, x)
            loss = sum(L.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if it % 500 == 0:
                self.log.append(dict(it=it, **{k: float(v) for k, v in L.items()}))
        if o.lbfgs_iters > 0:
            xf = self.sample(2 * o.n_colloc, d.t0, d.t1)
            lb = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
            state = self.rng.bit_generator.state

            def closure():
                lb.zero_grad()
                self.rng.bit_generator.state = state          # одни и те же граничные точки внутри шага
                ls = sum(self.losses(net, xf).values())
                ls.backward()
                return ls
            for _ in range(max(1, o.lbfgs_iters // 20)):
                lb.step(closure)
        self.log.append(dict(it=o.adam_iters + o.lbfgs_iters, **{k: float(v) for k, v in self.losses(net, x).items()}))
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.seconds = time.time() - t_start
        return net.cpu()


def relative_residual(tr: Trainer, net: PriceNet, t_hi: float, n=2000, avoid_days=4, seed=12345) -> float:
    """Самоконтроль: относительная невязка rms(R) / rms(|C_t| + |(r+s) C| + |f|) во внутренней части области
    (10-90% по каждой переменной, t в [0, t_hi]), вдали от купонных дат и даты колла. Так же её меряет проверка."""
    d = tr.dom
    rng = np.random.default_rng(seed)
    lo = np.array([0.0, d.r_lo, d.s_lo])
    hi = np.array([t_hi, d.r_hi, d.s_hi])
    X = lo + (hi - lo) * (0.1 + 0.8 * rng.random((4 * n, 3)))
    bad = np.r_[tr.sch.coupon_times, tr.sch.t_call]
    X = X[np.min(np.abs(X[:, :1] - bad[None, :]), axis=1) > avoid_days / 365][:n]
    net = net.to(tr.dev)
    for p in net.parameters():
        p.requires_grad_(False)
    Xt = tr.tens(X, True)
    R = tr.residual(net, Xt).detach().cpu().numpy().ravel()
    C = net(Xt)
    Ct = torch.autograd.grad(C.sum(), Xt)[0][:, 0].detach().cpu().numpy()
    C = C.detach().cpu().numpy().ravel()
    tt = Xt[:, 0].detach()
    f = (tr.bt.rate(tt) - (Xt[:, 1] + Xt[:, 2]).detach() * tr.bt.accrued(tt)).cpu().numpy()
    k = X[:, 1] + X[:, 2]
    scale = np.sqrt(np.mean((np.abs(Ct) + np.abs(k * C) + np.abs(f)) ** 2))
    net.cpu()
    return float(np.sqrt(np.mean(R ** 2)) / max(scale, 1e-12))


def predict(net: PriceNet, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net(torch.tensor(np.asarray(X, float), dtype=torch.float32)).numpy().reshape(-1).astype(float)
