"""Проверки загрузки: общий модуль локального checker и скрипта оценки платформы (один и тот же файл).

Что проверяется (раздел 4 постановки, README.md репозитория):
  * структура: predictions.csv, code.txt, pinn/<market>/{model.onnx, manifest.json} для каждого рынка;
  * code.txt: 1-я строка https-ссылка на репозиторий, 2-я строка хеш коммита (40 шестнадцатеричных символов);
  * формат predictions.csv: колонки, пары (рынок, дата), числа, диапазоны;
  * согласованность Z-спреда с ценой (пересчёт по фиксированной конвенции раздела 3.2);
  * PINN по каждому рынку: ONNX (вход x float32 [N, d], 2 <= d <= 6, выход [N, 1], размер, время),
    manifest.json (безопасный разбор выражений), невязка уравнения, корректность уравнения,
    переменные состояния внутри области, согласованность сети с ценами.
Истина здесь не используется: метрики считает только платформа.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import math
import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_limits(path=None) -> dict:
    return json.loads(Path(path or HERE / "limits.json").read_text(encoding="utf-8"))


class CheckError(Exception):
    """Ошибка, после которой проверка дальше не идёт (структура архива, нечитаемый CSV)."""


def where(market=None, row=None, field_=None) -> str:
    parts = []
    if market is not None:
        parts.append(f"рынок {market}")
    if row is not None:
        parts.append(f"строки {row}" if "," in str(row) else f"строка {row}")
    if field_ is not None:
        parts.append(f"поле {field_}")
    return "[" + ", ".join(parts) + "] " if parts else ""


# ---------------------------------------------------------------------------------------------------------------
# Данные рынка (только файлы папки рынка)
# ---------------------------------------------------------------------------------------------------------------
def _block(d, name):
    b = d.get(name)
    return pd.DataFrame(b["data"], columns=b["columns"]) if b else pd.DataFrame()


_A = [0.0, 0.6]
for _i in range(2, 9):
    _A.append(_A[-1] + 0.6 * 1.6 ** (_i - 1))
_BW = [_A[1]]
for _i in range(1, 9):
    _BW.append(_BW[-1] * 1.6)
G_COLS = ["B1", "B2", "B3", "T1"] + [f"G{i}" for i in range(1, 10)]


def gcurve_bp(p, tau):
    """G-кривая ЦБ (б.п., непрерывное начисление) по параметрам [B1, B2, B3, T1, G1..G9] (методика ЦБ)."""
    tau = np.asarray(tau, float)
    b0, b1, b2, t1 = p[:4]
    g = b0 + (b1 + b2) * t1 / tau * (1 - np.exp(-tau / t1)) - b2 * np.exp(-tau / t1)
    for k in range(9):
        g = g + p[4 + k] * np.exp(-((tau - _A[k]) ** 2) / _BW[k] ** 2)
    return g


class MarketData:
    """Целевая облигация рынка: купоны, колл, НКД по дням, параметры G-кривой."""

    def __init__(self, path):
        self.path = Path(path)
        self.code = self.path.name
        task = json.loads((self.path / "task.json").read_text(encoding="utf-8"))
        self.target = task["target"]
        self.eval_from, self.eval_till = pd.Timestamp(task["eval_from"]), pd.Timestamp(task["eval_till"])
        bz = json.loads((self.path / "bondization" / f"{self.target}.json").read_text(encoding="utf-8"))
        cp = _block(bz, "coupons")
        self.coupon_dates = pd.DatetimeIndex(pd.to_datetime(cp.coupondate))
        self.coupon_starts = pd.DatetimeIndex(pd.to_datetime(cp.startdate))
        desc = _block(json.loads((self.path / "securities" / f"{self.target}.json").read_text(encoding="utf-8")),
                      "description")
        d = dict(zip(desc.name, desc.value))
        self.face = float(d.get("FACEVALUE", 1000) or 1000)
        self.coupon_pct = cp.value.values.astype(float) / self.face * 100.0     # % номинала за период
        of = _block(bz, "offers")
        calls = sorted(pd.to_datetime(of[of.offertype == "Колл-опцион"].offerdate)) if len(of) else []
        self.maturity = pd.Timestamp(d.get("MATDATE")) if d.get("MATDATE") else self.coupon_dates[-1]
        self.call_date = pd.Timestamp(calls[0]) if calls else None
        h = _block(json.loads((self.path / "history" / f"{self.target}.json").read_text(encoding="utf-8")), "history")
        h["TRADEDATE"] = pd.to_datetime(h.TRADEDATE)
        self.hist = h
        self.accint_rub = pd.Series(h.ACCINT.values.astype(float), index=pd.DatetimeIndex(h.TRADEDATE))
        z = json.loads((self.path / "zcyc.json").read_text(encoding="utf-8"))
        p = _block(z, "params")
        p["tradedate"] = pd.to_datetime(p.tradedate)
        self.zparams = p.set_index("tradedate").sort_index()[G_COLS].astype(float)

    @property
    def end_date(self):
        """Дата последнего потока по конвенции Z-спреда: первый колл, иначе погашение."""
        return self.call_date if self.call_date is not None else self.maturity

    # -- Z-спред (раздел 3.2 постановки) -------------------------------------------------------------------------
    def zspread_bp(self, dates, clean) -> np.ndarray:
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        clean = np.asarray(clean, float)
        acc = self.accint_rub.reindex(dates).values / self.face * 100.0
        P = self.zparams.reindex(self.zparams.index.union(dates)).ffill().reindex(dates).values
        end = self.end_date
        rows_cf, rows_w, rows_x, tg, idx = [], [], [], [], []
        for k, d in enumerate(dates):
            if not (np.isfinite(clean[k]) and np.isfinite(acc[k]) and np.all(np.isfinite(P[k]))):
                continue
            m = (self.coupon_dates > d) & (self.coupon_dates <= end)
            if not m.any():
                continue
            tau = (self.coupon_dates[m] - d).days.values / 365.0
            cf = self.coupon_pct[m].copy()
            cf[-1] += 100.0
            if self.coupon_dates[m][-1] != end:          # колл не в купонную дату: номинал в дату колла
                tau = np.r_[tau, (end - d).days / 365.0]
                cf = np.r_[cf, 100.0]
            g = gcurve_bp(P[k], tau) / 1e4
            rows_cf.append(cf); rows_w.append(np.exp(-g * tau)); rows_x.append(tau)
            tg.append(clean[k] + acc[k]); idx.append(k)
        out = np.full(len(dates), np.nan)
        if idx:
            out[np.array(idx)] = _solve_rows(_pad(rows_cf), _pad(rows_w), _pad(rows_x), np.array(tg)) * 1e4
        return out

    # -- функции целевой облигации для manifest (в % номинала, ось t = (дата - origin) / 365) -------------------
    def bond_functions(self, origin: pd.Timestamp):
        ct = (self.coupon_dates - origin).days.values / 365.0
        st = (self.coupon_starts - origin).days.values / 365.0
        amt = self.coupon_pct

        def _idx(t):
            return np.clip(np.searchsorted(ct, t, side="right"), 0, len(ct) - 1)

        def coupon_rate(t):
            t = np.asarray(t, float)
            i = _idx(t)
            return amt[i] / (ct[i] - st[i])

        def accrued(t):
            t = np.asarray(t, float)
            i = _idx(t)
            a = amt[i] * (t - st[i]) / (ct[i] - st[i])
            return np.where((t < st[0]) | (t >= ct[-1]), 0.0, np.maximum(a, 0.0))
        return {"coupon_rate": coupon_rate, "accrued": accrued}, ct


def _solve_rows(CF, W, X, target, lo=-0.5, hi=1.0, iters=80, tol=1e-13):
    """sum_k CF_k W_k exp(-s X_k) = target по строкам: монотонная выпуклая функция, Ньютон с вилкой."""
    a = np.full(len(target), lo)
    b = np.full(len(target), hi)
    f = lambda s: np.sum(CF * W * np.exp(-s[:, None] * X), 1) - target
    ok = (f(a) > 0) & (f(b) < 0) & np.isfinite(target)
    s = np.where(ok, 0.05, np.nan)
    for _ in range(iters):
        e = np.exp(-s[:, None] * X)
        val = np.sum(CF * W * e, 1) - target
        der = -np.sum(CF * W * X * e, 1)
        a = np.where(val > 0, s, a)
        b = np.where(val <= 0, s, b)
        sn = s - val / np.where(der != 0, der, -1e-12)
        bad = ~((sn > a) & (sn < b))
        sn = np.where(bad, 0.5 * (a + b), sn)
        done = np.nanmax(np.abs(sn - s)) < tol if ok.any() else True
        s = sn
        if done:
            break
    return np.where(ok, s, np.nan)


def _pad(rows):
    w = max((len(r) for r in rows), default=1)
    out = np.zeros((len(rows), max(w, 1)))
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


# ---------------------------------------------------------------------------------------------------------------
# Безопасный разбор выражений manifest
# ---------------------------------------------------------------------------------------------------------------
FUNCS = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "min": np.minimum, "max": np.maximum}


class ManifestError(Exception):
    pass


class SafeExpr:
    """Арифметика: числа, объявленные имена, + - * / **, унарный минус, вызовы exp log sqrt abs min max,
    таблиц из `tables`, coupon_rate(t), accrued(t). Атрибуты, индексы, строки, лямбды, ключевые аргументы,
    импорты и любые другие вызовы запрещены (ast, без eval)."""

    ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call, ast.Add, ast.Sub,
               ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)

    def __init__(self, src, names: set, funcs: set, max_len: int, label: str):
        if not isinstance(src, (str, int, float)) or isinstance(src, bool):
            raise ManifestError(f"{label}: выражение должно быть строкой, получено {type(src).__name__}")
        src = str(src)
        if len(src) > max_len:
            raise ManifestError(f"{label}: выражение длиннее {max_len} символов ({len(src)})")
        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError as e:
            raise ManifestError(f"{label}: синтаксическая ошибка в {src[:80]!r}: {e.msg}")
        for node in ast.walk(tree):
            if not isinstance(node, self.ALLOWED):
                raise ManifestError(f"{label}: запрещённая конструкция {type(node).__name__} в {src[:80]!r}")
            if isinstance(node, ast.Constant) and (not isinstance(node.value, (int, float)) or isinstance(node.value, bool)):
                raise ManifestError(f"{label}: допустимы только числовые константы, найдено {node.value!r}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in funcs or node.keywords:
                    nm = ast.unparse(node.func)[:60]
                    raise ManifestError(f"{label}: вызов {nm!r} не разрешён (можно: exp, log, sqrt, abs, min, max, "
                                        f"таблицы из tables, coupon_rate, accrued; без именованных аргументов)")
            if isinstance(node, ast.Name) and node.id not in names and node.id not in funcs:
                raise ManifestError(f"{label}: неизвестное имя {node.id!r}")
        self.tree, self.src = tree, src
        self.names_used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def __call__(self, env, n):
        v = self._ev(self.tree.body, env)
        return np.broadcast_to(np.asarray(v, float), (n,)).astype(float)

    def _ev(self, nd, env):
        if isinstance(nd, ast.Constant):
            return float(nd.value)
        if isinstance(nd, ast.Name):
            return env[nd.id]
        if isinstance(nd, ast.UnaryOp):
            v = self._ev(nd.operand, env)
            return -v if isinstance(nd.op, ast.USub) else v
        if isinstance(nd, ast.BinOp):
            a, b = self._ev(nd.left, env), self._ev(nd.right, env)
            op = type(nd.op)
            if op is ast.Add:
                return a + b
            if op is ast.Sub:
                return a - b
            if op is ast.Mult:
                return a * b
            if op is ast.Div:
                return a / b
            if np.any(np.abs(np.asarray(b)) > 8):
                raise ManifestError("показатель степени по модулю больше 8")
            return np.power(a, b)
        if isinstance(nd, ast.Call):
            return env[nd.func.id](*[self._ev(x, env) for x in nd.args])
        raise ManifestError("недопустимое выражение")


@dataclass
class Manifest:
    origin: pd.Timestamp
    names: List[str]
    domain: np.ndarray
    drift: Dict[str, SafeExpr]
    diff: Dict[tuple, SafeExpr]
    disc: SafeExpr
    src: SafeExpr
    env_f: dict
    avoid_t: np.ndarray


def parse_manifest(path: Path, mkt: MarketData, lim: dict) -> Manifest:
    path = Path(path)
    if not path.exists():
        raise ManifestError("нет файла manifest.json")
    if path.stat().st_size > lim["manifest_max_kb"] * 1024:
        raise ManifestError(f"manifest.json больше {lim['manifest_max_kb']} КБ")
    try:
        m = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        raise ManifestError(f"manifest.json не является корректным JSON: {e}")
    if not isinstance(m, dict):
        raise ManifestError("manifest.json: ожидается JSON-объект")
    for k in ("time_origin", "variables", "pde"):
        if k not in m:
            raise ManifestError(f"manifest.json: нет обязательного ключа {k!r}")
    try:
        origin = pd.Timestamp(str(m["time_origin"]))
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(m["time_origin"])):
            raise ValueError
    except Exception:
        raise ManifestError(f"time_origin должен быть датой YYYY-MM-DD, получено {m['time_origin']!r}")
    vars_ = m["variables"]
    if not isinstance(vars_, list) or not (lim["d_min"] <= len(vars_) <= lim["d_max"]):
        raise ManifestError(f"variables: от {lim['d_min']} до {lim['d_max']} переменных, первая — время t")
    names, dom = [], []
    for i, v in enumerate(vars_):
        if not isinstance(v, dict) or not {"name", "lo", "hi"} <= set(v):
            raise ManifestError(f"variables[{i}]: нужны поля name, lo, hi")
        nm = v["name"]
        if not isinstance(nm, str) or not nm.isidentifier() or nm in FUNCS or nm in ("coupon_rate", "accrued"):
            raise ManifestError(f"variables[{i}]: недопустимое имя {nm!r}")
        try:
            lo, hi = float(v["lo"]), float(v["hi"])
        except Exception:
            raise ManifestError(f"variables[{i}] ({nm}): lo и hi должны быть числами")
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ManifestError(f"variables[{i}] ({nm}): нужно конечное lo < hi")
        names.append(nm); dom.append((lo, hi))
    if len(set(names)) != len(names):
        raise ManifestError("variables: имена переменных повторяются")
    if names[0] != "t":
        raise ManifestError("variables: первая переменная должна называться 't'")
    tables = {}
    tb_raw = m.get("tables") or {}
    if not isinstance(tb_raw, dict):
        raise ManifestError("tables: ожидается объект {имя: {x: [...], y: [...]}}")
    for k, tb in tb_raw.items():
        if not isinstance(k, str) or not k.isidentifier() or k in names or k in FUNCS or k in ("coupon_rate", "accrued"):
            raise ManifestError(f"tables: недопустимое имя таблицы {k!r}")
        try:
            x, y = np.asarray(tb["x"], float), np.asarray(tb["y"], float)
        except Exception:
            raise ManifestError(f"tables.{k}: x и y должны быть массивами чисел")
        if x.ndim != 1 or len(x) != len(y) or len(x) < 2 or len(x) > lim["table_max_len"] or np.any(np.diff(x) <= 0) \
                or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ManifestError(f"tables.{k}: x строго возрастает, len(x) = len(y), 2..{lim['table_max_len']} точек")
        tables[k] = (lambda xx, x=x, y=y: np.interp(xx, x, y))
    funcs = set(FUNCS) | set(tables) | {"coupon_rate", "accrued"}
    nm = set(names)
    pde = m["pde"]
    if not isinstance(pde, dict):
        raise ManifestError("pde: ожидается объект")
    for k in ("diffusion", "discount"):
        if k not in pde:
            raise ManifestError(f"pde: нет обязательного ключа {k!r}")
    L = lim["expr_max_len"]
    drift_raw = pde.get("drift") or {}
    if not isinstance(drift_raw, dict) or not isinstance(pde["diffusion"], dict):
        raise ManifestError("pde.drift и pde.diffusion: ожидаются объекты {переменная: выражение}")
    drift = {}
    for k, v in drift_raw.items():
        if k not in nm or k == "t":
            raise ManifestError(f"pde.drift: ключ {k!r} не является переменной состояния")
        drift[k] = SafeExpr(v, nm, funcs, L, f"pde.drift.{k}")
    diff = {}
    for k, v in pde["diffusion"].items():
        ab = [x.strip() for x in str(k).split(",")]
        if len(ab) != 2 or ab[0] not in nm or ab[1] not in nm or "t" in ab:
            raise ManifestError(f"pde.diffusion: ключ {k!r} должен иметь вид 'x,y' для переменных состояния")
        key = tuple(ab) if names.index(ab[0]) <= names.index(ab[1]) else (ab[1], ab[0])
        if key in diff:
            raise ManifestError(f"pde.diffusion: элемент {key} задан дважды")
        diff[key] = SafeExpr(v, nm, funcs, L, f"pde.diffusion.{k}")
    disc = SafeExpr(pde["discount"], nm, funcs, L, "pde.discount")
    src = SafeExpr(pde.get("source", "0"), nm, funcs, L, "pde.source")
    bf, ct = mkt.bond_functions(origin)
    avoid = list(ct)
    if mkt.call_date is not None:
        avoid.append((mkt.call_date - origin).days / 365.0)
    return Manifest(origin, names, np.array(dom, float), drift, diff, disc, src, dict(FUNCS, **tables, **bf),
                    np.array(avoid, float))


# ---------------------------------------------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------------------------------------------
class OnnxError(Exception):
    pass


class OnnxModel:
    def __init__(self, path: Path, d: int, lim: dict):
        path = Path(path)
        if not path.exists():
            raise OnnxError("нет файла model.onnx")
        mb = path.stat().st_size / 2 ** 20
        if mb > lim["onnx_max_mb"]:
            raise OnnxError(f"model.onnx {mb:.1f} МБ, предел {lim['onnx_max_mb']} МБ")
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(lim["onnx_threads"])
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        try:
            self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise OnnxError(f"model.onnx не загружается onnxruntime: {str(e).splitlines()[0][:200]}")
        ins, outs = self.sess.get_inputs(), self.sess.get_outputs()
        if len(ins) != 1 or ins[0].name != "x":
            raise OnnxError(f"нужен ровно один вход с именем 'x', найдено: {[i.name for i in ins]}")
        if ins[0].type != "tensor(float)":
            raise OnnxError(f"вход 'x' должен быть float32 (tensor(float)), найдено {ins[0].type}")
        shp = ins[0].shape
        if len(shp) != 2:
            raise OnnxError(f"вход 'x' должен быть двумерным [N, d], найдено {shp}")
        if isinstance(shp[1], int) and shp[1] != d:
            raise OnnxError(f"размерность входа d = {shp[1]}, а в manifest.json {d} переменных")
        if not (lim["d_min"] <= d <= lim["d_max"]):
            raise OnnxError(f"d = {d} вне [{lim['d_min']}; {lim['d_max']}]")
        if len(outs) < 1 or outs[0].type != "tensor(float)":
            raise OnnxError(f"выход должен быть float32, найдено {[o.type for o in outs]}")
        self.d = d
        self.out_name = outs[0].name

    def __call__(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = []
        for i in range(0, len(X), 100_000):
            try:
                y = self.sess.run([self.out_name], {"x": X[i:i + 100_000]})[0]
            except Exception as e:
                raise OnnxError(f"ошибка расчёта сети: {str(e).splitlines()[0][:200]}")
            if y.ndim != 2 or y.shape != (len(X[i:i + 100_000]), 1):
                raise OnnxError(f"выход должен иметь форму [N, 1], получено {list(y.shape)}")
            out.append(y.reshape(-1))
        return np.concatenate(out).astype(float) if out else np.zeros(0)

    def timing(self, man: Manifest, lim: dict) -> float:
        rng = np.random.default_rng(1)
        lo, hi = man.domain[:, 0], man.domain[:, 1]
        n = int(lim["onnx_points"])
        X = lo + (hi - lo) * rng.random((n, self.d))
        t0 = time.perf_counter()                       # оценка по 1000 точкам: заведомо медленную сеть не досчитываем
        self(X[:1000])
        est = (time.perf_counter() - t0) * n / 1000
        if est > 3 * lim["onnx_max_seconds"]:
            raise OnnxError(f"расчёт {n} точек занял бы около {est:.0f} с (оценка по 1000 точкам), "
                            f"предел {lim['onnx_max_seconds']:g} с")
        t0 = time.perf_counter()
        y = self(X)
        dt = time.perf_counter() - t0
        if not np.all(np.isfinite(y)):
            raise OnnxError("сеть выдаёт NaN или inf в точках области")
        return dt


# ---------------------------------------------------------------------------------------------------------------
# Невязка и корректность уравнения
# ---------------------------------------------------------------------------------------------------------------
def residual_check(fn, man: Manifest, lim: dict) -> dict:
    """Относительная невязка rms(R) / rms(|V_t| + |k V| + |f|) в случайных точках внутренней части области
    (10-90% по каждой переменной), вдали от купонных дат и даты колла; производные — центральные разности."""
    n = int(lim["residual_points"])
    rng = np.random.default_rng(int(lim["residual_seed"]))
    names, dom = man.names, man.domain
    d = len(names)
    lo, hi = dom[:, 0], dom[:, 1]
    span = hi - lo
    X = lo + 0.1 * span + 0.8 * span * rng.random((n * 4, d))
    if len(man.avoid_t):
        w = lim["residual_avoid_days"] / 365.0
        X = X[np.min(np.abs(X[:, :1] - man.avoid_t[None, :]), axis=1) > w]
    X = X[:n]
    if len(X) < n // 4:
        raise ManifestError("область по t почти целиком состоит из окрестностей купонных дат")
    h = 5e-3 * span
    h[0] = min(2.0 / 365.0, 0.05 * span[0])
    V = fn(X)
    d1, d2 = {}, {}
    for i in range(d):
        e = np.zeros(d); e[i] = h[i]
        Vp, Vm = fn(X + e), fn(X - e)
        d1[names[i]] = (Vp - Vm) / (2 * h[i])
        d2[(names[i], names[i])] = (Vp - 2 * V + Vm) / h[i] ** 2
    for (a, b) in man.diff:
        if a != b:
            i, j = names.index(a), names.index(b)
            ei = np.zeros(d); ei[i] = h[i]
            ej = np.zeros(d); ej[j] = h[j]
            d2[(a, b)] = (fn(X + ei + ej) - fn(X + ei - ej) - fn(X - ei + ej) + fn(X - ei - ej)) / (4 * h[i] * h[j])
    env = dict(man.env_f, **{nm: X[:, k] for k, nm in enumerate(names)})
    with np.errstate(all="ignore"):
        R = d1["t"].copy()
        for k, ex in man.drift.items():
            R += ex(env, len(X)) * d1[k]
        coef = {}
        for (a, b), ex in man.diff.items():
            coef[(a, b)] = ex(env, len(X))
            R += (0.5 if a == b else 1.0) * coef[(a, b)] * d2[(a, b)]
        k_ = man.disc(env, len(X))
        f = man.src(env, len(X))
        R += -k_ * V + f
        scale = np.sqrt(np.mean((np.abs(d1["t"]) + np.abs(k_ * V) + np.abs(f)) ** 2))
    out = dict(rel_residual=float(np.sqrt(np.mean(R ** 2)) / max(scale, 1e-12)) if np.all(np.isfinite(R)) else float("nan"),
               n_points=int(len(X)))
    # корректность уравнения
    errs = []
    states = names[1:]
    if not np.all(np.isfinite(k_)):
        errs.append("дисконт k даёт NaN/inf в точках области")
    used = man.disc.names_used & set(states)
    if not used:
        errs.append("дисконт k не зависит ни от одной переменной состояния (ставки, спреда)")
    elif np.nanstd(k_) < 1e-10:
        errs.append("дисконт k численно постоянен по области")
    S = np.zeros((len(X), len(states), len(states)))
    for (a, b), c in coef.items():
        i, j = states.index(a), states.index(b)
        S[:, i, j] = c
        S[:, j, i] = c
    if not np.all(np.isfinite(S)):
        errs.append("коэффициенты диффузии дают NaN/inf в точках области")
    else:
        diag = np.einsum("nii->ni", S)
        if np.max(np.abs(diag)) <= 0:
            errs.append("диффузия нулевая: уравнение вырождено")
        ev = np.linalg.eigvalsh(S).min(axis=1)
        tol = 1e-9 * max(np.max(np.abs(S)), 1e-30)
        if np.any(ev < -tol):
            errs.append(f"матрица диффузии не является неотрицательно определённой (в {int(np.sum(ev < -tol))} точках)")
    out["equation_errors"] = errs
    return out


# ---------------------------------------------------------------------------------------------------------------
# Архив и структура загрузки
# ---------------------------------------------------------------------------------------------------------------
def safe_extract(zip_path: Path, dest: Path, lim: dict) -> Path:
    """Распаковка с защитой: не zip / битый архив, пути с '..' и абсолютные, бомбы (объём, степень сжатия)."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise CheckError(f"файл {zip_path} не найден")
    if not zipfile.is_zipfile(zip_path):
        raise CheckError("загрузка не является zip-архивом")
    try:
        z = zipfile.ZipFile(zip_path)
        infos = z.infolist()
    except Exception as e:
        raise CheckError(f"битый zip-архив: {e}")
    if len(infos) > lim["zip_max_files"]:
        raise CheckError(f"в архиве {len(infos)} файлов, предел {lim['zip_max_files']}")
    total = 0
    for i in infos:
        n = i.filename.replace("\\", "/")
        if n.startswith("/") or re.match(r"^[A-Za-z]:", n) or ".." in n.split("/"):
            raise CheckError(f"недопустимый путь в архиве: {i.filename!r}")
        total += i.file_size
        if i.compress_size > 0 and i.file_size / i.compress_size > lim["zip_max_ratio"] and i.file_size > 2 ** 20:
            raise CheckError(f"подозрительно высокая степень сжатия файла {i.filename!r} (архив-бомба?)")
    if total > lim["zip_max_total_mb"] * 2 ** 20:
        raise CheckError(f"распакованный объём {total / 2 ** 20:.0f} МБ больше {lim['zip_max_total_mb']} МБ")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        for i in infos:
            if i.is_dir():
                continue
            target = dest / i.filename.replace("\\", "/")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as dst:
                left = i.file_size + 1
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    left -= len(buf)
                    if left < 0:
                        raise CheckError(f"файл {i.filename!r} больше заявленного размера")
                    dst.write(buf)
    except CheckError:
        raise
    except Exception as e:
        raise CheckError(f"битый zip-архив: {e}")
    return dest


def _junk(p: Path) -> bool:
    return p.name.startswith(".") or p.name == "__MACOSX" or p.name.startswith("._")


def find_root(sub_dir: Path) -> Path:
    """Корень загрузки: папка с predictions.csv; допускается одна вложенная корневая папка; __MACOSX/.DS_Store
    и прочие лишние файлы игнорируются."""
    sub_dir = Path(sub_dir)
    if (sub_dir / "predictions.csv").exists():
        return sub_dir
    subs = [p for p in sub_dir.iterdir() if p.is_dir() and not _junk(p)] if sub_dir.exists() else []
    if len(subs) == 1 and (subs[0] / "predictions.csv").exists():
        return subs[0]
    raise CheckError("в архиве нет predictions.csv (в корне или в единственной вложенной папке)")


def check_code_txt(root: Path):
    p = root / "code.txt"
    if not p.exists():
        raise CheckError("в архиве нет code.txt (две строки: https-ссылка на репозиторий и хеш коммита)")
    lines = [x.strip() for x in p.read_text(encoding="utf-8-sig", errors="replace").splitlines() if x.strip()]
    if len(lines) < 2:
        raise CheckError("code.txt: нужны две непустые строки — https-ссылка на репозиторий и хеш коммита")
    url, sha = lines[0], lines[1]
    if not re.fullmatch(r"https://[^\s/]+\.[^\s/]+/\S+", url):
        raise CheckError(f"code.txt, строка 1: нужна https-ссылка на репозиторий, получено {url[:100]!r}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise CheckError(f"code.txt, строка 2: нужен хеш коммита из 40 шестнадцатеричных символов, получено {sha[:60]!r}")
    return url, sha.lower()


# ---------------------------------------------------------------------------------------------------------------
# predictions.csv
# ---------------------------------------------------------------------------------------------------------------
REQUIRED = ["market", "date", "clean_price", "zspread_bp"]


def read_predictions(path: Path, lim: dict) -> pd.DataFrame:
    path = Path(path)
    if path.stat().st_size > lim["csv_max_mb"] * 2 ** 20:
        raise CheckError(f"predictions.csv больше {lim['csv_max_mb']} МБ")
    raw = path.read_bytes()
    try:
        txt = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CheckError("predictions.csv: файл должен быть в кодировке UTF-8")
    head = txt.splitlines()[0] if txt.strip() else ""
    if not head:
        raise CheckError("predictions.csv пуст")
    if ";" in head and "," not in head:
        raise CheckError("predictions.csv: разделитель колонок должен быть запятой ',' (найдена ';'), "
                         "десятичный знак — точка")
    try:
        df = pd.read_csv(io.StringIO(txt), dtype=str, keep_default_na=False, skipinitialspace=True)
    except Exception as e:
        raise CheckError(f"predictions.csv не читается как CSV: {str(e).splitlines()[0][:200]}")
    df.columns = [c.strip() for c in df.columns]
    return df


def _num(s: pd.Series):
    return pd.to_numeric(s.str.strip(), errors="coerce")


def check_table(pred: pd.DataFrame, tmpl: pd.DataFrame, lim: dict, errors: list) -> Optional[pd.DataFrame]:
    """Колонки, пары (рынок, дата), числа и диапазоны. Возвращает таблицу в порядке шаблона (или None)."""
    miss = [c for c in REQUIRED if c not in pred.columns]
    if miss:
        errors.append(f"predictions.csv: нет обязательной колонки {', '.join(miss)} "
                      f"(нужны {', '.join(REQUIRED)}; split и z_required — из шаблона)")
        return None
    if pred.columns.duplicated().any():
        errors.append(f"predictions.csv: повторяющиеся названия колонок {sorted(set(pred.columns[pred.columns.duplicated()]))}")
        return None
    pred = pred.copy()
    pred["lineno"] = np.arange(len(pred)) + 2           # номер строки файла (заголовок — строка 1)
    pred["market"] = pred.market.str.strip()
    pred["date"] = pred.date.str.strip()
    dup = pred[pred.duplicated(["market", "date"], keep=False)]
    for (mk, dt), g in list(dup.groupby(["market", "date"]))[:10]:
        errors.append(f"{where(mk, ', '.join(map(str, g.lineno)))}дубликат пары (рынок, дата) {dt}")
    if len(dup):
        return None
    key_t = set(zip(tmpl.market, tmpl.date))
    key_p = set(zip(pred.market, pred.date))
    lost = sorted(key_t - key_p)
    extra = pred[[k not in key_t for k in zip(pred.market, pred.date)]]
    for mk, dt in lost[:10]:
        errors.append(f"{where(mk)}нет строки для даты {dt}")
    if len(lost) > 10:
        errors.append(f"... всего пропущено пар (рынок, дата): {len(lost)}")
    for r in extra.head(10).itertuples():
        errors.append(f"{where(r.market, r.lineno)}лишняя пара (рынок, дата) {r.date}: её нет в шаблоне")
    if len(extra) > 10:
        errors.append(f"... всего лишних строк: {len(extra)}")
    if lost or len(extra):
        return None
    mg = tmpl[["market", "date", "split", "z_required"]].merge(
        pred.drop(columns=[c for c in ("split", "z_required") if c in pred.columns]), on=["market", "date"], how="left")
    for c in ("split", "z_required"):
        if c in pred.columns:
            chk = tmpl[["market", "date", c]].merge(pred[["market", "date", c, "lineno"]], on=["market", "date"],
                                                     suffixes=("", "_p"))
            bad = chk[chk[c].astype(str).str.strip() != chk[c + "_p"].astype(str).str.strip()]
            for r in bad.head(5).itertuples():
                errors.append(f"{where(r.market, r.lineno, c)}значение изменено относительно шаблона")
    n_bad = 0
    for c in ("clean_price", "zspread_bp"):
        raw = mg[c].astype(str).str.strip()
        v = _num(raw)
        need = np.ones(len(mg), bool) if c == "clean_price" else (mg.z_required.values == 1)
        empty_ok = (~need) & (raw.values == "")
        bad = (~np.isfinite(v.values)) & ~empty_ok
        comma = raw.str.match(r"^-?\d+,\d+$")
        for i in np.where(bad)[0][:10]:
            hint = " (десятичный знак должен быть точкой)" if comma.iloc[i] else ""
            errors.append(f"{where(mg.market.iloc[i], int(mg.lineno.iloc[i]), c)}нечисловое, пустое или бесконечное "
                          f"значение {raw.iloc[i][:30]!r}{hint}")
        n_bad += int(bad.sum())
        if bad.sum() > 10:
            errors.append(f"... всего некорректных значений в {c}: {int(bad.sum())}")
        mg[c] = v.where(~empty_ok, np.nan)
    if n_bad:
        return None
    pr = mg.clean_price.values
    out = (pr < lim["price_min"]) | (pr > lim["price_max"])
    for i in np.where(out)[0][:10]:
        errors.append(f"{where(mg.market.iloc[i], int(mg.lineno.iloc[i]), 'clean_price')}цена {pr[i]:g} вне "
                      f"[{lim['price_min']:g}; {lim['price_max']:g}]")
    zr = mg.z_required.values == 1
    zb = zr & (np.abs(mg.zspread_bp.values) > lim["zspread_abs_max_bp"])
    for i in np.where(zb)[0][:10]:
        errors.append(f"{where(mg.market.iloc[i], int(mg.lineno.iloc[i]), 'zspread_bp')}|Z-спред| = "
                      f"{abs(mg.zspread_bp.values[i]):g} больше {lim['zspread_abs_max_bp']:g} б.п.")
    if out.any() or zb.any():
        return None
    return mg


def check_zspread(mg: pd.DataFrame, markets: Dict[str, MarketData], lim: dict, errors: list):
    """Пересчёт Z из цены; расхождение > z_tol_bp допускается не более чем в (1 - z_share_ok) строк с z_required = 1."""
    bad_rows, tot = [], 0
    for code, g in mg[mg.z_required == 1].groupby("market", sort=False):
        z = markets[code].zspread_bp(g.date.values, g.clean_price.values)
        diff = np.abs(z - g.zspread_bp.values)
        bad = ~(diff <= lim["z_tol_bp"])
        tot += len(g)
        for i in np.where(bad)[0]:
            bad_rows.append((code, int(g.lineno.values[i]), g.zspread_bp.values[i], z[i]))
    share = 1 - len(bad_rows) / max(tot, 1)
    if share < lim["z_share_ok"]:
        errors.append(f"Z-спред не согласован с ценой: расхождение больше {lim['z_tol_bp']:g} б.п. в {len(bad_rows)} "
                      f"из {tot} строк (допустимо не более {100 * (1 - lim['z_share_ok']):.0f}%). Примеры:")
        for code, line, zs, zc in bad_rows[:8]:
            errors.append(f"  {where(code, line, 'zspread_bp')}сдано {zs:.3f}, пересчитано из цены {zc:.3f}")
    return dict(z_rows=tot, z_mismatch=len(bad_rows))


# ---------------------------------------------------------------------------------------------------------------
# PINN по рынку
# ---------------------------------------------------------------------------------------------------------------
def check_pinn_market(code: str, mdir: Path, mkt: MarketData, rows: Optional[pd.DataFrame], lim: dict) -> dict:
    res = dict(market=code, passed=False, rel_residual=float("nan"), consistency_max=float("nan"), onnx_seconds=float("nan"),
               d=None, errors=[])
    E = res["errors"]
    try:
        man = parse_manifest(mdir / "manifest.json", mkt, lim)
        res["d"] = len(man.names)
        net = OnnxModel(mdir / "model.onnx", len(man.names), lim)
        res["onnx_seconds"] = net.timing(man, lim)
        if res["onnx_seconds"] > lim["onnx_max_seconds"]:
            E.append(f"расчёт {lim['onnx_points']} точек занял {res['onnx_seconds']:.1f} с, предел {lim['onnx_max_seconds']:g} с")
        rc = residual_check(net, man, lim)
        res["rel_residual"] = rc["rel_residual"]
        if not (rc["rel_residual"] < lim["residual_max"]):
            E.append(f"невязка уравнения {rc['rel_residual']:.4f} не ниже порога {lim['residual_max']:g}")
        E.extend(rc["equation_errors"])
        if rows is not None:
            t = (pd.to_datetime(rows.date) - man.origin).dt.days.values / 365.0
            cols = [t]
            for k, nm in enumerate(man.names[1:], start=1):
                if nm not in rows.columns:
                    E.append(f"в predictions.csv нет колонки состояния {nm!r} из manifest.json")
                    return res
                v = _num(rows[nm].astype(str))
                badv = ~np.isfinite(v.values)
                if badv.any():
                    i = int(np.where(badv)[0][0])
                    E.append(f"{where(None, int(rows.lineno.values[i]), nm)}нечисловое значение состояния")
                    return res
                cols.append(v.values)
            X = np.stack(cols, 1)
            lo, hi = man.domain[:, 0], man.domain[:, 1]
            outside = (X < lo - 1e-12) | (X > hi + 1e-12)
            for j in range(X.shape[1]):
                if outside[:, j].any():
                    i = int(np.where(outside[:, j])[0][0])
                    E.append(f"{where(None, int(rows.lineno.values[i]), man.names[j])}значение {X[i, j]:.6g} вне области "
                             f"[{lo[j]:.6g}; {hi[j]:.6g}] из manifest.json (всего {int(outside[:, j].sum())} строк)")
            if not outside.any():
                y = net(X)
                dev = np.abs(y - rows.clean_price.values.astype(float))
                res["consistency_max"] = float(np.max(dev)) if np.all(np.isfinite(dev)) else float("inf")
                if not (res["consistency_max"] <= lim["consistency_tol"]):
                    i = int(np.nanargmax(np.where(np.isfinite(dev), dev, np.inf)))
                    E.append(f"сеть в ваших состояниях не воспроизводит цены: максимум расхождения "
                             f"{res['consistency_max']:.4f} п.п. > {lim['consistency_tol']:g} "
                             f"(строка {int(rows.lineno.values[i])}: сеть {y[i]:.4f}, цена {rows.clean_price.values[i]:.4f})")
    except (ManifestError, OnnxError) as e:
        E.append(str(e))
    except Exception as e:                                   # неожиданная ошибка не роняет проверку
        E.append(f"ошибка проверки: {type(e).__name__}: {str(e)[:200]}")
    res["passed"] = not E
    return res


# ---------------------------------------------------------------------------------------------------------------
# Вся загрузка
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Report:
    ok: bool = False
    format_ok: bool = False
    pinn_ok: bool = False
    errors: List[str] = field(default_factory=list)
    pinn: List[dict] = field(default_factory=list)
    code_url: str = ""
    code_sha: str = ""
    table: Optional[pd.DataFrame] = None
    z_info: dict = field(default_factory=dict)
    seconds: float = 0.0
    markets: dict = field(default_factory=dict)

    @property
    def pinn_pass(self):
        return float(np.mean([p["passed"] for p in self.pinn])) if self.pinn else 0.0

    @property
    def pinn_residual_median(self):
        v = [p["rel_residual"] for p in self.pinn if np.isfinite(p["rel_residual"])]
        return float(np.median(v)) if v else float("nan")


def load_template(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, dtype={"market": str, "date": str, "split": str})
    t["z_required"] = t.z_required.astype(int)
    return t


def check_submission(sub_dir: Path, data_dir: Path, lim: dict, template: Optional[pd.DataFrame] = None) -> Report:
    """sub_dir: распакованная загрузка; data_dir: папка data/ (markets/<code>/..., predictions_template.csv)."""
    t0 = time.time()
    rep = Report()
    data_dir = Path(data_dir)
    tmpl = template if template is not None else load_template(data_dir / "predictions_template.csv")
    codes = list(dict.fromkeys(tmpl.market))
    try:
        root = find_root(Path(sub_dir))
        rep.code_url, rep.code_sha = check_code_txt(root)
        if not (root / "pinn").is_dir():
            raise CheckError("в архиве нет папки pinn/ (pinn/<market>/model.onnx и manifest.json для каждого рынка)")
        missing = [c for c in codes if not (root / "pinn" / c).is_dir()]
        if missing:
            raise CheckError(f"нет папки pinn/<market>/ для {len(missing)} рынков из {len(codes)}: "
                             f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
        pred = read_predictions(root / "predictions.csv", lim)
    except CheckError as e:
        rep.errors.append(str(e))
        rep.seconds = time.time() - t0
        return rep
    markets = {c: MarketData(data_dir / "markets" / c) for c in codes}
    rep.markets = markets
    mg = check_table(pred, tmpl, lim, rep.errors)
    if mg is not None:
        rep.z_info = check_zspread(mg, markets, lim, rep.errors)
    rep.format_ok = mg is not None and not rep.errors
    rep.table = mg
    for c in codes:
        rows = mg[mg.market == c] if mg is not None else None
        rep.pinn.append(check_pinn_market(c, root / "pinn" / c, markets[c], rows, lim))
    rep.pinn_ok = all(p["passed"] for p in rep.pinn)
    rep.ok = rep.format_ok and rep.pinn_ok
    rep.seconds = time.time() - t0
    return rep


def format_report(rep: Report, maxlinenos=60) -> str:
    out = []
    if rep.ok:
        out.append("ПРИНЯТО: загрузка прошла все проверки формата и PINN.")
    else:
        out.append("ОТКЛОНЕНО: загрузка не прошла проверку.")
    if rep.errors:
        out.append("Формат:")
        out += ["  " + e for e in rep.errors[:maxlinenos]]
    if rep.pinn:
        n_ok = sum(p["passed"] for p in rep.pinn)
        out.append(f"PINN: прошли {n_ok} из {len(rep.pinn)} рынков; медиана невязки {rep.pinn_residual_median:.4f}")
        for p in rep.pinn:
            if not p["passed"]:
                for e in p["errors"][:5]:
                    out.append(f"  [рынок {p['market']}] {e}")
    if rep.code_url:
        out.append(f"Код: {rep.code_url} @ {rep.code_sha}")
    out.append(f"Время проверки: {rep.seconds:.1f} с")
    return "\n".join(out)
