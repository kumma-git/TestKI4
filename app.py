from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="KI-Bilderkennung",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stApp { max-width: 800px; margin: 0 auto; }
    .title-text { text-align: center; font-weight: 700; color: #1e293b; }
    .subtitle-text { text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 30px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Modell wird geladen …")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH.name}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_labels():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels nicht gefunden: {LABELS_PATH.name}")

    labels = []
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Teachable-Machine-Labels sind oft z. B. "0 Kleidungsstücke".
        number, separator, label = line.partition(" ")
        labels.append(label.strip() if separator and number.isdigit() else line)
    return labels


def prepare_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.fit(image.convert("RGB"), IMAGE_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    return np.expand_dims((pixels / 127.5) - 1.0, axis=0)


def to_probabilities(output: np.ndarray) -> np.ndarray:
    scores = np.asarray(output, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Das Modell hat ungültige Werte zurückgegeben.")

    # Falls das Modell Logits liefert, in Wahrscheinlichkeiten umwandeln.
    if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=1e-3):
        scores = tf.nn.softmax(scores).numpy()
    return np.clip(scores, 0.0, 1.0)


st.markdown("<h1 class='title-text'>🧠 KI-Bilderkennung</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle-text'>Lade ein Bild hoch und sehe die Vorhersagen als Prozentwerte.</p>",
    unsafe_allow_html=True,
)

try:
    model = load_model()
    labels = load_labels()
except Exception as error:
    st.error(f"Fehler beim Laden: {error}")
    st.info("Lege app.py, keras_model.h5 und labels.txt in denselben Ordner.")
    st.stop()

uploaded_file = st.file_uploader(
    "Bild auswählen", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    with st.spinner("Analyse läuft …"):
        probabilities = to_probabilities(model.predict(prepare_image(image), verbose=0)[0])

    if len(labels) != len(probabilities):
        st.error(
            f"Anzahl der Labels ({len(labels)}) passt nicht zur Modell-Ausgabe "
            f"({len(probabilities)} Klassen)."
        )
        st.stop()

    results = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
    top_label, top_probability = results[0]

    st.divider()
    st.subheader("📊 Erkennungsergebnis")
    st.success(f"Hauptvorhersage: **{top_label} — {top_probability * 100:.1f}%**")

    st.write("Alle Klassen:")
    for label, probability in results:
        percentage = float(probability * 100)
        left, right = st.columns([1, 2])
        with left:
            st.markdown(f"**{label}**")
            st.caption(f"{percentage:.2f}%")
        with right:
            st.progress(float(probability), text=f"{percentage:.1f}%")
