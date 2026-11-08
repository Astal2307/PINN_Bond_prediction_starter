"""Экспорт сети рынка: pinn/<market>/model.onnx (вход x float32 [N, 3] = (t, r, s), выход [N, 1]) и manifest.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import Model
from .pinn import Domain, PriceNet


def export_onnx(net: PriceNet, path: Path):
    net = net.eval().cpu()
    torch.onnx.export(net, torch.zeros(4, 3), str(path), input_names=["x"], output_names=["price"],
                      dynamic_axes={"x": {0: "N"}, "price": {0: "N"}}, opset_version=17, dynamo=False)


def manifest(model: Model, dom: Domain, time_origin: str, t_hi: float) -> dict:
    return {"time_origin": time_origin,
            "variables": [{"name": "t", "lo": 0.0, "hi": float(t_hi)},
                          {"name": "r", "lo": float(dom.r_lo), "hi": float(dom.r_hi), "meaning": "short rate (Vasicek)"},
                          {"name": "s", "lo": float(dom.s_lo), "hi": float(dom.s_hi), "meaning": "credit spread (constant)"}],
            "pde": model.manifest_pde(),
            "tables": {}}


def write_market(out_dir: Path, net: PriceNet, model: Model, dom: Domain, time_origin: str, t_hi: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(net, out_dir / "model.onnx")
    (out_dir / "manifest.json").write_text(json.dumps(manifest(model, dom, time_origin, t_hi), indent=1), encoding="utf-8")


def onnx_predict(path: Path, X: np.ndarray) -> np.ndarray:
    """Цены из самого ONNX-файла: так predictions.csv точно согласован с сетью."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return s.run(None, {"x": np.asarray(X, np.float32)})[0].reshape(-1).astype(float)
