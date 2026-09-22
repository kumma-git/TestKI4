"""Teachable-Machine Bilderkennung für die Fundbox (TestKI4) — ohne TensorFlow.

Die Gewichte aus ``keras_model.h5`` werden mit purem numpy gerechnet
(``keras_numpy.py``, MobileNetV2-Forward-Pass). Läuft überall, wo es
numpy + h5py gibt — auch auf Python 3.14 ohne TF-Wheels.
Klassen: ``labels.txt``.
"""

from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Quellen (TestKI4-Original) — das Modell selbst lädt keras_numpy.
# --------------------------------------------------------------------------
TESTKI4_LABELS_URL = (
    "https://raw.githubusercontent.com/kumma-git/TestKI4/main/labels.txt"
)
# TestKI4 nennt die Datei labels.txt (Format "0 Klassenname").
LABEL_FILES = [Path("teachable_labels.txt"), Path("labels.txt")]

IMAGE_SIZE = 224

# --------------------------------------------------------------------------
# Mapping: Teachable-Klasse -> Fundbox-Kategorie (1:1, gleiche Namen).
# Die 11 Klassen aus labels.txt SIND die Kategorien der Fundbox.
# --------------------------------------------------------------------------
TEACHABLE_TO_CATEGORY = {name: name for name in [
    "Kleidungsstücke",
    "Schulsachen",
    "Trinkflasche",
    "Brotdose",
    "Regenschirm",
    "Schlüssel",
    "Kopfhörer",
    "Powerbank/Ladekabel",
    "Brille",
    "Geldtasche",
    "Taschenrechner",
]}

# Fallback, falls labels.txt fehlt: Reihenfolge aus TestKI4.
BUILTIN_LABELS = list(TEACHABLE_TO_CATEGORY)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
def parse_teachable_labels(lines: list[str]) -> list[str]:
    """Parst TestKI4-Labels im Format ``"<index> <Name>"``.

    Robust gegen Varianten ohne Index, Leerzeilen und Mehrwort-Namen.
    """
    names: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].rstrip(".:").isdigit():
            names.append(parts[1].strip())
        else:
            names.append(line)
    return names


def ensure_file(path: Path, url: str) -> Path | None:
    """Lädt eine Datei bei Bedarf herunter (einmalig)."""
    if path.exists() and path.stat().st_size > 1000:
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    except Exception:
        return None
    if path.exists() and path.stat().st_size > 1000:
        return path
    return None


def load_teachable_labels() -> list[str]:
    for candidate in LABEL_FILES:
        if candidate.exists() and candidate.stat().st_size > 0:
            try:
                parsed = parse_teachable_labels(
                    candidate.read_text(encoding="utf-8").splitlines()
                )
                if parsed:
                    return parsed
            except Exception:
                continue
    path = ensure_file(LABEL_FILES[0], TESTKI4_LABELS_URL)
    if path is not None:
        try:
            return parse_teachable_labels(
                path.read_text(encoding="utf-8").splitlines()
            ) or list(BUILTIN_LABELS)
        except Exception:
            pass
    return list(BUILTIN_LABELS)


# --------------------------------------------------------------------------
# Modell (numpy-Engine auf keras_model.h5, gecached) — kein TensorFlow nötig
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_teachable_graph():
    """Netzwerk-Graph + Gewichte oder None (Datei fehlt / h5py fehlt)."""
    try:
        import keras_numpy  # noqa: F401
    except Exception:
        return None
    import keras_numpy

    try:
        return keras_numpy.load_graph()
    except Exception:
        return None


def preprocess(pil_image: Image.Image) -> np.ndarray:
    """224x224 RGB, Teachable-Normalisierung (/127.5 - 1), Batch-Dim."""
    img = pil_image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    arr = np.asarray(img, dtype=np.float32)
    data = np.ndarray(shape=(1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    data[0] = (arr / 127.5) - 1.0
    return data


def predict_teachable(pil_image: Image.Image) -> dict | None:
    """Gibt ``{label, confidence, category, top3, engine}`` oder None zurück."""
    import keras_numpy

    graph = load_teachable_graph()
    if graph is None:
        return None
    try:
        labels = load_teachable_labels()
        probs = keras_numpy.forward_batch(graph, preprocess(pil_image))[0]
        probs = np.asarray(probs, dtype=np.float64).ravel()
        n = min(len(probs), len(labels))
        probs, labels = probs[:n], labels[:n]
        order = np.argsort(probs)[::-1]
        top3 = [(labels[int(i)], float(probs[int(i)])) for i in order[:3]]
        best_label, best_prob = top3[0]
        return {
            "label": best_label,
            "confidence": best_prob,
            "category": TEACHABLE_TO_CATEGORY.get(best_label, "Sonstiges"),
            "top3": top3,
            "engine": "Teachable Machine (keras_model.h5 · numpy)",
        }
    except Exception:
        return None


# --------------------------------------------------------------------------
# Heuristik-Fallback (kein Modell verfügbar) — ehrlich als unsicher markiert
# --------------------------------------------------------------------------
def heuristic_guess(pil_image: Image.Image) -> dict:
    rgb = pil_image.convert("RGB")
    w, h = rgb.size
    small = np.asarray(rgb.resize((64, 64)), dtype=np.float32)
    std = float(small.std(axis=(0, 1)).mean())
    r, g, b = (float(v) for v in small.mean(axis=(0, 1)))
    if std < 18 and r < 80 and g < 80 and b < 80:
        return {
            "label": "Dunkles Objekt",
            "confidence": 0.35,
            "category": "Kopfhörer",
            "top3": [],
            "engine": "Bildmerkmale (Fallback, unsicher)",
        }
    if w / float(h) < 0.62 or w / float(h) > 1.7:
        return {
            "label": "Längliches Objekt",
            "confidence": 0.35,
            "category": "Trinkflasche",
            "top3": [],
            "engine": "Bildmerkmale (Fallback, unsicher)",
        }
    return {
        "label": "Unbekannt",
        "confidence": 0.25,
        "category": "Kleidungsstücke",
        "top3": [],
        "engine": "Kein Modell verfügbar",
    }
