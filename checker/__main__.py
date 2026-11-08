"""Локальная проверка загрузки (те же проверки, что на платформе, без истины).

  python -m checker submission.zip --data data/
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

from .core import CheckError, Report, check_submission, format_report, load_limits, safe_extract


def run(zip_path, data_dir, limits=None) -> Report:
    lim = limits or load_limits()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            safe_extract(Path(zip_path), Path(tmp), lim)
        except CheckError as e:
            rep = Report()
            rep.errors.append(str(e))
            return rep
        return check_submission(Path(tmp), Path(data_dir), lim)


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Проверка submission.zip перед загрузкой на платформу")
    ap.add_argument("zip", help="путь к submission.zip")
    ap.add_argument("--data", default="data", help="папка data/ репозитория")
    ap.add_argument("--details", action="store_true", help="таблица проверки PINN по всем рынкам")
    a = ap.parse_args(argv)
    rep = run(a.zip, a.data)
    print(format_report(rep))
    if a.details and rep.pinn:
        print(f"{'рынок':10s} {'PINN':6s} {'невязка':>8s} {'согл.':>8s} {'ONNX, с':>8s}")
        for p in rep.pinn:
            print(f"{p['market']:10s} {'да' if p['passed'] else 'НЕТ':6s} {p['rel_residual']:8.4f} "
                  f"{p['consistency_max']:8.4f} {p['onnx_seconds']:8.3f}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
