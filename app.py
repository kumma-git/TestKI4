import streamlit as st
import os
import json
import uuid
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import tensorflow as tf


# ============================================================
# KONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fundbox",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

APP_NAME = "Fundbox"

# Datenordner
DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
ITEMS_FILE = DATA_DIR / "items.json"
MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

CATEGORIES = {
    "Kleidungsstücke": "👕",
    "Zubehör": "🧴",
    "Schulsachen": "📖",
    "Sonstiges": "❓"
}


# ============================================================
# DATENBANK - EINFACHES JSON
# ============================================================

def load_items():
    """Lädt alle gefundenen Gegenstände."""
    if not ITEMS_FILE.exists():
        return []

    try:
        with open(ITEMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_items(items):
    """Speichert alle gefundenen Gegenstände."""
    with open(ITEMS_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def add_item(item):
    """Fügt einen Gegenstand hinzu."""
    items = load_items()
    items.insert(0, item)
    save_items(items)


# ============================================================
# KERAS / TENSORFLOW MODELL LADEN
# ============================================================

@st.cache_resource
def load_keras_model():
    """Lädt das Keras-Modell einmalig in den Speicher."""
    if not MODEL_PATH.exists():
        return None
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Keras-Modells: {e}")
        return None


def load_labels():
    """Lädt die Klassen-Labels (falls vorhanden)."""
    if not LABELS_PATH.exists():
        return []
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        return labels
    except Exception:
        return []


# ============================================================
# KI: BILD ERKENNEN MIT KERAS MODEL
# ============================================================

def recognize_item(image_bytes):
    """
    Erkennt den Gegenstand auf einem Foto mithilfe von keras_model.h5.
    """
    model = load_keras_model()
    if model is None:
        return None, "Das Keras-Modell ('keras_model.h5') konnte nicht geladen werden oder fehlt im Hauptverzeichnis."

    try:
        # Bild öffnen und vorbereiten (224x224 RGB wie vom Teachable Machine / Keras Standard gefordert)
        image = Image.open(image_bytes).convert("RGB")
        image = image.resize((224, 224))
        
        # In NumPy-Array umwandeln & Normalisieren (-1 bis 1 bzw. 0 bis 1 je nach Modellstandard)
        image_array = np.asarray(image, dtype=np.float32)
        normalized_image_array = (image_array / 127.5) - 1.0  # Normalisierung für Teachable Machine
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Vorhersage durchführen
        prediction = model.predict(data)
        index = int(np.argmax(prediction[0]))
        
        labels = load_labels()
        
        # Name anhand der Labels oder des Class-Index bestimmen
        if labels and index < len(labels):
            # Oft steht in labels.txt z.B. "0 Trinkflasche", daher teilen wir am Leerzeichen
            raw_label = labels[index]
            predicted_name = raw_label.split(" ", 1)[-1] if " " in raw_label else raw_label
        else:
            predicted_name = f"Klasse {index}"

        # Zuordnung zu den Kategorien der Fundbox
        name_lower = predicted_name.lower()
        if any(w in name_lower for w in ["pullover", "shirt", "jacke", "hose", "mütze", "schal", "schuh", "kleidung"]):
            category = "Kleidungsstücke"
        elif any(w in name_lower for w in ["flasche", "dose", "schlüssel", "schirm", "brille", "kopfhörer"]):
            category = "Zubehör"
        elif any(w in name_lower for w in ["heft", "buch", "ordner", "federmappe", "rechner", "stift", "lineal"]):
            category = "Schulsachen"
        else:
            category = "Sonstiges"

        result = {
            "name": predicted_name.capitalize(),
            "category": category,
            "description": f"Automatisch erkanntes Objekt ({predicted_name}).",
            "search_terms": [predicted_name.lower(), category.lower()]
        }

        return result, None

    except Exception as e:
        return None, f"Fehler bei der Bildanalyse: {e}"


# ============================================================
# SUCH-FUNKTION (VOLLTEXT-SUCHE)
# ============================================================

def simple_search(query, items):
    """
    Durchsucht den Datenbestand nach Suchbegriffen.
    """
    query = query.lower().strip()

    if not query:
        return items

    words = query.split()
    scored = []

    for item in items:
        text = " ".join([
            item.get("name", ""),
            item.get("category", ""),
            item.get("description", ""),
            " ".join(item.get("search_terms", []))
        ]).lower()

        score = 0
        for word in words:
            if word in text:
                score += 1

        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]


# ============================================================
# CSS - DESIGN WIE AUF DER SKIZZE
# ============================================================

st.markdown(
    """
<style>

    /* Gesamte Seite */
    .stApp {
        background: #ffffff;
    }

    .main .block-container {
        max-width: 920px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }


    /* Fundbox */
    .fundbox-title {
        font-family: "Comic Sans MS", "Trebuchet MS", sans-serif;
        font-size: 54px;
        font-weight: 400;
        color: #222222;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }


    /* Suchfeld */
    div[data-testid="stTextInput"] input {
        border: 2px solid #9a9a9a !important;
        border-radius: 35px !important;
        padding: 16px 25px !important;
        font-size: 18px !important;
        background: white !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #9b9b9b !important;
        opacity: 1 !important;
    }


    /* Überschriften */
    .section-title {
        font-family: "Comic Sans MS", "Trebuchet MS", sans-serif;
        font-size: 30px;
        color: #222222;
        margin-top: 45px;
        margin-bottom: 22px;
    }


    /* Kategorien */
    .category-card {
        border: 2px solid #333333;
        border-radius: 25px;
        padding: 18px 8px;
        text-align: center;
        min-height: 125px;
        background: #ffffff;
    }

    .category-icon {
        font-size: 54px;
        line-height: 1.1;
    }

    .category-name {
        font-family: "Comic Sans MS", "Trebuchet MS", sans-serif;
        font-size: 17px;
        margin-top: 8px;
        color: #333333;
    }


    /* Gegenstände */
    .item-card {
        border: 2px solid #333333;
        border-radius: 25px;
        padding: 10px;
        background: white;
        height: 100%;
    }

    .item-name {
        font-family: "Comic Sans MS", "Trebuchet MS", sans-serif;
        text-align: center;
        font-size: 17px;
        margin-top: 8px;
    }

    .item-category {
        text-align: center;
        color: #777777;
        font-size: 13px;
    }


    /* Upload */
    .upload-box {
        border: 3px dashed #777777;
        border-radius: 25px;
        padding: 25px;
        background: #fafafa;
        text-align: center;
    }

    .upload-title {
        font-family: "Comic Sans MS", "Trebuchet MS", sans-serif;
        font-size: 22px;
    }


    /* Hinweise */
    .hints {
        font-family: "Comic Sans MS", "Trebuchet MS", sans-serif;
        color: #777777;
        font-size: 14px;
        line-height: 1.7;
        margin-top: 10px;
    }


    /* Buttons */
    .stButton > button {
        border: 2px solid #444444;
        border-radius: 20px;
        background: white;
        color: #222222;
        font-size: 15px;
    }

    .stButton > button:hover {
        border-color: #111111;
        color: #111111;
    }


    /* Trennlinie */
    .sketch-line {
        border-top: 2px solid #444444;
        margin: 35px 0;
    }


    /* Logo */
    .school-logo {
        border: 2px solid #333333;
        border-radius: 50%;
        width: 130px;
        height: 130px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px auto;
        font-family: Georgia, serif;
        font-size: 45px;
        color: #222222;
    }

    .school-name {
        text-align: center;
        font-family: Georgia, serif;
        font-size: 15px;
        color: #444444;
    }


    /* Info */
    .info-box {
        padding: 15px;
        border-radius: 15px;
        background: #f5f5f5;
        border: 1px solid #dddddd;
    }

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "category_filter" not in st.session_state:
    st.session_state.category_filter = None

if "search_results" not in st.session_state:
    st.session_state.search_results = None

if "search_query" not in st.session_state:
    st.session_state.search_query = ""


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="fundbox-title">Fundbox</div>',
    unsafe_allow_html=True
)


# ============================================================
# SUCHFELD
# ============================================================

search_query = st.text_input(
    "Suchen",
    value=st.session_state.search_query,
    placeholder="Suchen",
    label_visibility="collapsed",
    key="search_input"
)

if search_query != st.session_state.search_query:
    st.session_state.search_query = search_query
    st.session_state.category_filter = None

    if search_query.strip():
        items = load_items()
        st.session_state.search_results = simple_search(search_query, items)
    else:
        st.session_state.search_results = None


# ============================================================
# KATEGORIEN
# ============================================================

st.markdown(
    '<div class="section-title">Kategorie:</div>',
    unsafe_allow_html=True
)

category_columns = st.columns(4)

for index, (category, icon) in enumerate(CATEGORIES.items()):

    with category_columns[index]:

        st.markdown(
            f"""
            <div class="category-card">
                <div class="category-icon">{icon}</div>
                <div class="category-name">{category}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            category,
            key=f"category_{category}",
            use_container_width=True
        ):
            if st.session_state.category_filter == category:
                st.session_state.category_filter = None
            else:
                st.session_state.category_filter = category

            st.session_state.search_results = None
            st.rerun()


# ============================================================
# GEGENSTÄNDE AUSWÄHLEN
# ============================================================

all_items = load_items()

if st.session_state.search_results is not None:
    visible_items = st.session_state.search_results
    section_text = "Suchergebnisse:"
elif st.session_state.category_filter:
    visible_items = [
        item
        for item in all_items
        if item.get("category") == st.session_state.category_filter
    ]
    section_text = st.session_state.category_filter + ":"
else:
    visible_items = all_items
    section_text = "Kürzlich gefunden:"


# ============================================================
# KÜRZLICH GEFUNDEN
# ============================================================

st.markdown(
    f'<div class="section-title">{section_text}</div>',
    unsafe_allow_html=True
)

if not visible_items:

    st.markdown(
        """
        <div class="info-box">
            Noch keine passenden Gegenstände gefunden.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    # Maximal 8 aktuelle Gegenstände auf der Startseite
    display_items = visible_items[:8]

    for row_start in range(0, len(display_items), 4):

        row_items = display_items[row_start:row_start + 4]
        cols = st.columns(4)

        for index, item in enumerate(row_items):

            with cols[index]:

                image_path = IMAGE_DIR / item["filename"]

                if image_path.exists():

                    st.markdown(
                        '<div class="item-card">',
                        unsafe_allow_html=True
                    )

                    st.image(
                        str(image_path),
                        use_container_width=True
                    )

                    st.markdown(
                        f"""
                        <div class="item-name">
                            {item["name"]}
                        </div>

                        <div class="item-category">
                            {item["category"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

                else:
                    st.warning("Bild nicht gefunden.")


# ============================================================
# ETWAS GEFUNDEN?
# ============================================================

st.markdown(
    '<div class="section-title">Etwas gefunden?</div>',
    unsafe_allow_html=True
)

upload_left, upload_right = st.columns(
    [1.4, 0.8],
    gap="large"
)


# ============================================================
# UPLOAD
# ============================================================

with upload_left:

    st.markdown(
        """
        <div class="upload-box">
            <div class="upload-title">
                📷 Foto hochladen
            </div>
            <br>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        label_visibility="collapsed",
        key="photo_upload"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hints">
            <b>Hinweise zum Foto:</b><br>
            • Gute Belichtung<br>
            • Gegenstand möglichst vollständig fotografieren<br>
            • Nicht zu weit entfernt fotografieren<br>
            • Möglichst ruhiger Hintergrund<br>
            • Keine Personen auf dem Foto
        </div>
        """,
        unsafe_allow_html=True
    )

    if uploaded_file:

        st.markdown("### Vorschau")

        image_bytes = uploaded_file.getvalue()

        st.image(
            image_bytes,
            use_container_width=True
        )

        if st.button(
            "🔎 Gegenstand erkennen und hinzufügen",
            use_container_width=True
        ):

            with st.spinner(
                "Keras-Modell erkennt den Gegenstand ..."
            ):

                result, error = recognize_item(uploaded_file)

            if error:

                st.error(error)

            elif result:

                name = result.get("name", "Unbekannter Gegenstand")
                category = result.get("category", "Sonstiges")
                description = result.get("description", "")
                search_terms = result.get("search_terms", [])

                if category not in CATEGORIES:
                    category = "Sonstiges"

                item_id = str(uuid.uuid4())
                original_name = uploaded_file.name

                extension = Path(original_name).suffix.lower()

                if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
                    extension = ".jpg"

                filename = item_id + extension
                image_path = IMAGE_DIR / filename

                # Bild speichern
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Gegenstand speichern
                new_item = {
                    "id": item_id,
                    "name": name,
                    "category": category,
                    "description": description,
                    "search_terms": search_terms,
                    "filename": filename,
                    "created_at": datetime.now().isoformat(timespec="seconds")
                }

                add_item(new_item)

                st.success(
                    f"Erkannt: {name} → {category}"
                )

                st.info(
                    "Der Gegenstand wurde zum Katalog "
                    "und zu 'Kürzlich gefunden' hinzugefügt."
                )

                st.rerun()


# ============================================================
# SCHULLOGO
# ============================================================

with upload_right:

    st.markdown(
        """
        <div style="height: 25px;"></div>

        <div class="school-logo">
            ✚
        </div>

        <div class="school-name">
            KATHARINEUM<br>
            ZU LÜBECK
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.caption(
        "Schul-App für verlorene und gefundene Gegenstände"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top: 70px;
        text-align: center;
        color: #999999;
        font-family: Arial, sans-serif;
        font-size: 12px;
    ">
        Fundbox · Katharineum zu Lübeck
    </div>
    """,
    unsafe_allow_html=True
)
