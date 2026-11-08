"""Шаг 7. Итоговый график по реальной облигации (раздел 3.5 постановки)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def real_bond_plot(days, clean, z_bp, s_model, obs: pd.DataFrame, call_date, title, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    if len(obs):
        sz = 10 + 120 * np.sqrt(obs.volume.values / max(obs.volume.max(), 1))
        a1.scatter(obs.date, obs.price, s=sz, alpha=0.5, color="tab:gray", label="сделки (WAPRICE, размер — объём)")
    a1.plot(days, clean, color="tab:blue", lw=1.6, label="справедливая цена (PINN)")
    for a in (a1, a2):
        if call_date is not None:
            a.axvline(call_date, color="tab:red", ls="--", lw=1)
    if call_date is not None:
        a1.text(call_date, a1.get_ylim()[1], " колл", color="tab:red", va="top")
    a1.set_ylabel("чистая цена, % номинала")
    a1.set_title(title)
    a1.legend(loc="lower right", fontsize=8)
    a2.plot(days, z_bp, color="tab:green", lw=1.3, label="Z-спред к G-кривой, б.п.")
    a2.plot(days, np.asarray(s_model) * 1e4, color="tab:purple", lw=1, ls=":", label="спред модели s, б.п.")
    a2.set_ylabel("б.п.")
    a2.legend(loc="upper left", fontsize=8)
    lo, hi = np.nanpercentile(np.r_[z_bp, np.asarray(s_model) * 1e4], [2, 98])
    a2.set_ylim(lo - 0.2 * (hi - lo + 1), hi + 0.2 * (hi - lo + 1))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
