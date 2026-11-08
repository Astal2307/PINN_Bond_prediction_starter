"""Бейзлайн целиком: папки рынков -> predictions.csv, pinn/<market>/, code.txt, submission.zip, real_bond.png.

  python -m baseline.run --data data/ --out out/

Рынки обучаются параллельно: на CPU — по процессу на ядро (один поток torch на процесс), на GPU (CUDA / Apple MPS) —
4 процесса на одном устройстве (--jobs меняет число). --device auto: CUDA, если есть, иначе CPU; Apple MPS — флагом.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


# самоконтроль: если относительная невязка сети выше RESIDUAL_TARGET (порог платформы 0.05 — с запасом),
# сеть переобучается с другим seed; берётся лучшая из MAX_ATTEMPTS попыток
RESIDUAL_TARGET = 0.035
MAX_ATTEMPTS = 3


def pick_device(name: str) -> str:
    import torch
    if name == "auto":                     # Apple MPS на сети бейзлайна медленнее CPU (замер в README) — только флагом
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA недоступна: используйте --device cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        sys.exit("Apple MPS недоступен: используйте --device cpu")
    return name


def _init_worker():
    import torch
    torch.set_num_threads(1)
    import warnings
    warnings.filterwarnings("ignore")


def market_job(args):
    """Шаги 1-6 для одного рынка. Возвращает строки predictions и сводку."""
    mdir, out_pinn, device, opts, seed = args
    from .data import Market
    from .calibrate import calibrate
    from .pinn import Trainer, TrainOpts, make_domain, relative_residual
    from .inverse import naive_states
    from .export import write_market, onnx_predict
    from .zspread import zspread_bp
    t0 = time.time()
    mkt = Market(mdir)                                                     # шаг 1
    model, st, info = calibrate(mkt)                                       # шаги 2-3
    sch = st["sch"]
    dom = make_domain(model, sch.future(), st["r"], st["s_range"])
    t_hi = min(sch.t_call, sch.T)
    t_hi = max(t_hi, float(st["t"][-1]) + 1 / 365)
    best, attempts, train_sec = None, [], 0.0
    for k in range(MAX_ATTEMPTS):                                          # шаг 4 (+ самоконтроль невязки)
        tr = Trainer(model, sch, dom, TrainOpts(seed=seed + 1000 * k, **opts), device=device)
        net = tr.train()
        train_sec += tr.seconds
        res = relative_residual(tr, net, t_hi)
        attempts.append(round(res, 4))
        if best is None or res < best[0]:
            best = (res, net)
        if res <= RESIDUAL_TARGET:
            break
    net = best[1]
    s, s_obs = naive_states(net, dom, st["t"], st["r"], st["obs"], model.s_level)   # шаг 5
    out_dir = Path(out_pinn) / mkt.code
    write_market(out_dir, net, model, dom, mkt.eval_from.strftime("%Y-%m-%d"), t_hi)
    X = np.c_[st["t"], st["r"], s]
    C = onnx_predict(out_dir / "model.onnx", X)
    z = zspread_bp(mkt, mkt.eval_days, C)                                  # шаг 6
    rows = pd.DataFrame(dict(market=mkt.code, date=mkt.eval_days.strftime("%Y-%m-%d"), clean_price=C, zspread_bp=z,
                             r=st["r"], s=s))
    info.update(market=mkt.code, train_seconds=train_sec, seconds=time.time() - t0, device=device,
                self_residual=best[0], attempts=len(attempts), residuals=str(attempts), final_loss=tr.log[-1])
    return rows, info, dict(obs=st["obs"], call=sch.t_call)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Бейзлайн: Васичек + постоянный спред + PINN + наивное обращение")
    ap.add_argument("--data", default="data", help="папка data/ репозитория")
    ap.add_argument("--out", default="out", help="куда писать результаты")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--jobs", type=int, default=0, help="число процессов (0: все ядра CPU; на GPU — 4)")
    ap.add_argument("--markets", default="", help="через запятую: только эти рынки (для отладки; zip не собирается)")
    ap.add_argument("--adam-iters", type=int, default=3000)
    ap.add_argument("--lbfgs-iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-real", action="store_true", help="не считать реальную облигацию")
    ap.add_argument("--repo-url", default=None, help="https-ссылка на ваш репозиторий (если origin не https)")
    a = ap.parse_args(argv)
    t_all = time.time()
    data, out = Path(a.data), Path(a.out)
    device = pick_device(a.device)
    jobs = a.jobs or (4 if device != "cpu" else os.cpu_count() or 1)
    tmpl = pd.read_csv(data / "predictions_template.csv", dtype={"market": str, "date": str})
    codes = list(dict.fromkeys(tmpl.market))
    if a.markets:
        codes = [c for c in codes if c in a.markets.split(",")]
    if out.exists():
        shutil.rmtree(out)
    (out / "pinn").mkdir(parents=True)
    opts = dict(adam_iters=a.adam_iters, lbfgs_iters=a.lbfgs_iters)
    seed_of = lambda c: a.seed + zlib.crc32(c.encode()) % 100000      # фиксированный seed для каждого рынка
    tasks = [(str(data / "markets" / c), str(out / "pinn"), device, opts, seed_of(c)) for c in codes]
    real_dir = out / "real"
    if not a.no_real and (data / "real").exists():
        tasks.append((str(data / "real"), str(real_dir), device, opts, seed_of("real")))
    print(f"рынков: {len(codes)}{' + реальная облигация' if len(tasks) > len(codes) else ''}; устройство {device}; "
          f"процессов {jobs}", flush=True)
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(min(jobs, len(tasks)), initializer=_init_worker) as pool:
        for k, res in enumerate(pool.imap_unordered(market_job, tasks)):
            results.append(res)
            i = res[1]
            print(f"  [{k + 1}/{len(tasks)}] {i['market']}: {i['seconds']:.0f} с, сделок в окне {i['n_trades_window']}, "
                  f"невязка {i['self_residual']:.4f}" + (f" (попыток {i['attempts']})" if i['attempts'] > 1 else ""),
                  flush=True)
    by = {r[1]["market"]: r for r in results}
    rows = pd.concat([by[c][0] for c in codes], ignore_index=True)
    info = pd.DataFrame([by[c][1] for c in codes])
    info.drop(columns=["final_loss"]).to_csv(out / "markets_info.csv", index=False)
    write_ok = True
    from .submission import write_predictions, code_info, make_zip, CodeInfoError
    full = len(codes) == tmpl.market.nunique()
    if full:
        write_predictions(rows, tmpl, out / "predictions.csv")
    else:
        rows.to_csv(out / "predictions_partial.csv", index=False)
    if "real" in by:
        rr, ri, extra = by["real"]
        from .data import Market
        from .plot import real_bond_plot
        m = Market(data / "real")
        call = m.tb.call_dates[0] if m.tb.call_dates else None
        rr.to_csv(out / "real_prices.csv", index=False)
        real_bond_plot(pd.to_datetime(rr.date), rr.clean_price.values, rr.zspread_bp.values, rr.s.values, extra["obs"],
                       call, f"{m.tb.desc.get('SHORTNAME', m.target)} ({m.target}): бейзлайн", out / "real_bond.png")
        print(f"график реальной облигации: {out / 'real_bond.png'}")
    if full:
        try:
            url, sha, dirty = code_info(Path.cwd(), a.repo_url)
            (out / "code.txt").write_text(f"{url}\n{sha}\n", encoding="utf-8")
            if dirty:
                print("ВНИМАНИЕ: есть незакоммиченные изменения; в code.txt записан хеш HEAD — закоммитьте код.")
            zp = make_zip(out)
            print(f"загрузка: {zp} ({zp.stat().st_size / 2 ** 20:.1f} МБ)")
        except CodeInfoError as e:
            write_ok = False
            print(f"ОШИБКА: submission.zip не собран: {e}", file=sys.stderr)
    print(f"готово за {time.time() - t_all:.0f} с", flush=True)
    return 0 if write_ok else 2


if __name__ == "__main__":
    sys.exit(main())
