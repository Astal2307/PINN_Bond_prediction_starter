"""Сборка загрузки: predictions.csv, code.txt (ссылка и хеш коммита из текущего git-репозитория), submission.zip."""
from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


class CodeInfoError(Exception):
    pass


HINT = ("Загрузка требует code.txt: https-ссылку на репозиторий с вашим кодом и хеш коммита.\n"
        "  1) создайте свой репозиторий (например, на GitHub) и отправьте туда код: \n"
        "       git remote set-url origin https://github.com/<team>/<repo>.git && git push origin HEAD\n"
        "  2) закоммитьте изменения (хеш берётся из HEAD), затем повторите запуск;\n"
        "  или укажите ссылку явно: python -m baseline.run ... --repo-url https://github.com/<team>/<repo>")


def _git(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=30)


def to_https(url: str) -> str:
    url = url.strip()
    m = re.fullmatch(r"git@([^:]+):(.+?)(\.git)?", url)
    if m:
        return f"https://{m.group(1)}/{m.group(2)}"
    m = re.fullmatch(r"ssh://git@([^/]+)/(.+?)(\.git)?", url)
    if m:
        return f"https://{m.group(1)}/{m.group(2)}"
    m = re.fullmatch(r"(https://.+?)(\.git)?", url)
    return m.group(1) if m else url


def code_info(cwd: Path, repo_url: str | None = None):
    try:
        top = _git(["rev-parse", "--show-toplevel"], cwd)
    except FileNotFoundError:
        raise CodeInfoError("git не установлен.\n" + HINT)
    if top.returncode != 0:
        raise CodeInfoError(f"{cwd} не является git-репозиторием.\n" + HINT)
    sha = _git(["rev-parse", "HEAD"], cwd).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise CodeInfoError("в репозитории нет коммитов.\n" + HINT)
    url = repo_url or _git(["remote", "get-url", "origin"], cwd).stdout.strip()
    url = to_https(url) if url else ""
    if not re.fullmatch(r"https://[^\s/]+\.[^\s/]+/\S+", url):
        raise CodeInfoError(f"ссылка на репозиторий {url or '(origin не задан)'!r} не является https-адресом.\n" + HINT)
    dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"], cwd).stdout.strip())
    return url, sha, dirty


def write_predictions(rows: pd.DataFrame, template: pd.DataFrame, path: Path):
    cols = ["market", "date", "clean_price", "zspread_bp", "split", "z_required"]
    t = template[["market", "date", "split", "z_required"]]
    df = t.merge(rows, on=["market", "date"], how="left")
    extra = [c for c in rows.columns if c not in cols]
    df = df[cols + extra]
    if df.clean_price.isna().any():
        raise RuntimeError(f"нет цены для {int(df.clean_price.isna().sum())} строк шаблона")
    df.to_csv(path, index=False, float_format="%.10g")
    return df


def make_zip(out: Path, zip_name="submission.zip"):
    zp = out / zip_name
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "predictions.csv", "predictions.csv")
        z.write(out / "code.txt", "code.txt")
        for p in sorted((out / "pinn").rglob("*")):
            if p.is_file():
                z.write(p, str(p.relative_to(out)))
    return zp
