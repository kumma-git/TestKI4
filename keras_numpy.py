"""Reine numpy-Inferenz für das Teachable-Machine-Modell (MobileNetV2).

Liest Architektur aus ``model_config`` und Gewichte aus ``model_weights``
der Datei ``keras_model.h5`` — ganz ohne TensorFlow (läuft überall, wo es
numpy + h5py gibt, z. B. Streamlit Cloud mit Python 3.14).

Unterstützte Layer: InputLayer, ZeroPadding2D, Conv2D, DepthwiseConv2D,
BatchNormalization (Inferenz), ReLU (auch ReLU6), Add,
GlobalAveragePooling2D, Dense, Activation (softmax/relu/linear).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MODEL_FILE = Path("keras_model.h5")
TESTKI4_MODEL_URL = (
    "https://raw.githubusercontent.com/kumma-git/TestKI4/main/keras_model.h5"
)


def _ensure_h5() -> Path | None:
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 1000:
        return MODEL_FILE
    try:
        import urllib.request

        urllib.request.urlretrieve(TESTKI4_MODEL_URL, MODEL_FILE)
    except Exception:
        return None
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 1000:
        return MODEL_FILE
    return None


# --------------------------------------------------------------------------
# Basis-Operationen (channels_last, float32)
# --------------------------------------------------------------------------
def _same_pads(in_size: int, stride: int, k: int) -> tuple[int, int]:
    out = -(-in_size // stride)
    pad = max((out - 1) * stride + k - in_size, 0)
    return pad // 2, pad - pad // 2


def _pad_same(x: np.ndarray, k: int, stride: int) -> np.ndarray:
    pt, pb = _same_pads(x.shape[0], stride[0], k)
    pl, pr = _same_pads(x.shape[1], stride[1], k)
    if pt + pb + pl + pr == 0:
        return x
    return np.pad(x, ((pt, pb), (pl, pr), (0, 0)))


def _conv_valid(x: np.ndarray, kernel: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
    """Kreuzkorrelation (wie TF-Conv), valid, mit Stride."""
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride
    patches = sliding_window_view(x, (kh, kw), axis=(0, 1))
    patches = patches[::sh, ::sw]
    if kernel.ndim == 4:  # (kh, kw, C, F)
        return np.tensordot(patches, kernel, axes=([2, 3, 4], [2, 0, 1]))
    raise ValueError(f"Unerwartete Kernel-Form {kernel.shape}")


def _conv2d(x, cfg, w) -> np.ndarray:
    k = w["kernel"]
    stride = tuple(cfg.get("strides", [1, 1]))
    if cfg.get("padding", "valid") == "same":
        x = _pad_same(x, k.shape[0], stride)
    return _conv_valid(x, k, stride)


def _depthwise2d(x, cfg, w) -> np.ndarray:
    k = w["kernel"]  # (kh, kw, C, depth_multiplier)
    stride = tuple(cfg.get("strides", [1, 1]))
    if cfg.get("padding", "valid") == "same":
        x = _pad_same(x, k.shape[0], stride)
    kh, kw = k.shape[0], k.shape[1]
    patches = sliding_window_view(x, (kh, kw), axis=(0, 1))
    patches = patches[:: stride[0], :: stride[1]]
    dm = k.shape[3]
    if dm == 1:
        k3 = np.ascontiguousarray(k[:, :, :, 0].transpose(2, 0, 1))  # (C, kh, kw)
        out = (patches * k3[None, None]).sum(axis=(3, 4))
        return np.ascontiguousarray(out)
    oh, ow, c = patches.shape[0], patches.shape[1], patches.shape[2]
    out = np.zeros((oh, ow, c * dm), dtype=np.float32)
    for m in range(dm):
        km = np.ascontiguousarray(k[:, :, :, m].transpose(2, 0, 1))
        out[:, :, m::dm] = (patches * km[None, None]).sum(axis=(3, 4))
    return out


def _batchnorm(x, cfg, w) -> np.ndarray:
    eps = float(cfg.get("epsilon", 0.001))
    return (x - w["mean"]) * w["gamma"] / np.sqrt(w["var"] + eps) + w["beta"]


def _relu(x, cfg) -> np.ndarray:
    out = np.maximum(x.astype(np.float32), 0.0)
    mv = cfg.get("max_value")
    if mv is not None:
        out = np.minimum(out, float(mv))
    return out


def _dense(x, w) -> np.ndarray:
    return x @ w["kernel"] + w["bias"]


def _activation(x, cfg) -> np.ndarray:
    act = (cfg.get("activation") or "linear").lower()
    if act == "relu":
        return np.maximum(x, 0.0)
    if act == "softmax":
        v = x - np.max(x)
        e = np.exp(v)
        return e / (np.sum(e) + 1e-12)
    return x


# --------------------------------------------------------------------------
# Graph laden
# --------------------------------------------------------------------------
def _find_layers(cfg: dict) -> tuple[list, list, list]:
    """Gibt (backbone_ops, gap_cfg, head_dense_cfgs) zurück."""
    top = cfg["config"]["layers"]
    seq1 = next(l for l in top if l["config"].get("name") == "sequential_1")
    inner = seq1["config"]["layers"]
    backbone = next(l for l in inner if l["class_name"] == "Functional")
    gap = next(l for l in inner if l["class_name"] == "GlobalAveragePooling2D")
    seq3 = next(l for l in top if l["config"].get("name") == "sequential_3")
    head = [l for l in seq3["config"]["layers"] if l["class_name"] == "Dense"]
    return backbone["config"]["layers"], gap["config"], [h["config"] for h in head]


def _inbound_names(layer: dict) -> list[str]:
    names: list[str] = []
    for node in layer.get("inbound_nodes", []) or []:
        for conn in node:
            if isinstance(conn, (list, tuple)) and conn:
                names.append(conn[0])
    return names


def _read_weights(h5: h5py.File) -> dict:
    """{(scope, layer, param): array} — float32."""
    out: dict = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            parts = name.split("/")
            # relativ zu model_weights: <scope>/<layer>/<param:0>
            if len(parts) == 3:
                scope, layer, param = parts
                param = param.split(":")[0]
                key = {"kernel": "kernel", "depthwise_kernel": "kernel",
                       "bias": "bias", "gamma": "gamma", "beta": "beta",
                       "moving_mean": "mean", "moving_variance": "var"}.get(param)
                if key:
                    out[(scope, layer, key)] = np.asarray(obj, dtype=np.float32)
    h5["model_weights"].visititems(visit)
    return out


@lru_cache(maxsize=1)
def load_graph() -> dict | None:
    """Architektur + Gewichte, oder None wenn Datei fehlt."""
    path = _ensure_h5()
    if path is None:
        return None
    try:
        with h5py.File(path, "r") as h5:
            model_cfg = json.loads(h5.attrs["model_config"])
            weights = _read_weights(h5)
        ops, gap_cfg, head = _find_layers(model_cfg)
        return {"ops": ops, "gap": gap_cfg, "head": head, "weights": weights}
    except Exception:
        return None


# --------------------------------------------------------------------------
# Forward-Pass
# --------------------------------------------------------------------------
def forward_batch(graph: dict, batch: np.ndarray) -> np.ndarray:
    """batch: (N, 224, 224, 3) float32 in [-1, 1] -> (N, 11) Wahrscheinlichkeiten."""
    weights = graph["weights"]
    out = np.stack([_forward_single(graph, weights, batch[i]) for i in range(batch.shape[0])])
    return out


def _forward_single(graph: dict, weights: dict, x: np.ndarray) -> np.ndarray:
    tensors: dict[str, np.ndarray] = {}
    for layer in graph["ops"]:
        cls = layer["class_name"]
        cfg = layer["config"]
        name = cfg.get("name", "")
        if cls == "InputLayer":
            tensors[name] = x
            continue
        src = _inbound_names(layer)
        if cls == "Add":
            tensors[name] = tensors[src[0]] + tensors[src[1]]
            continue
        a = tensors[src[0]]
        if cls == "ZeroPadding2D":
            (pt, pb), (pl, pr) = cfg["padding"]
            tensors[name] = np.pad(a, ((pt, pb), (pl, pr), (0, 0)))
        elif cls == "Conv2D":
            w = {"kernel": weights[("sequential_1", name, "kernel")]}
            tensors[name] = _conv2d(a, cfg, w).astype(np.float32)
        elif cls == "DepthwiseConv2D":
            w = {"kernel": weights[("sequential_1", name, "kernel")]}
            tensors[name] = _depthwise2d(a, cfg, w).astype(np.float32)
        elif cls == "BatchNormalization":
            w = {k: weights[("sequential_1", name, k)] for k in ("gamma", "beta", "mean", "var")}
            tensors[name] = _batchnorm(a, cfg, w).astype(np.float32)
        elif cls == "ReLU":
            tensors[name] = _relu(a, cfg)
        elif cls == "Activation":
            tensors[name] = _activation(a, cfg)
        elif cls == "Dropout":
            tensors[name] = a
        else:
            raise ValueError(f"Layer-Typ nicht unterstützt: {cls} ({name})")

    last = graph["ops"][-1]["config"].get("name", "")
    feat = tensors[last].mean(axis=(0, 1))  # GlobalAveragePooling2D
    for h in graph["head"]:
        wk = weights[("sequential_3", h["name"], "kernel")]
        wb = weights.get(("sequential_3", h["name"], "bias"))
        feat = feat @ wk + (wb if wb is not None else 0.0)
        feat = _activation(feat, h)
    return feat.astype(np.float64)
