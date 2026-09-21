import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# ---------------------------------------------------------
# Page Configuration & Custom CSS (Modern Design)
# ---------------------------------------------------------
st.set_page_config(
    page_title="KI Bildklassifizierung",
    page_icon="🧠",
    layout="centered"
)

# Modernes UI-Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .title-text {
        text-align: center;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 5px;
    }
    .subtitle-text {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions (Model & Labels Loading)
# ---------------------------------------------------------
@st.cache_resource
def load_keras_model():
    # Lädt das Keras-Modell aus dem selben Ordner
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    return model

@st.cache_data
def load_labels():
    # Lädt die Labels aus labels.txt
    with open("labels.txt", "r", encoding="utf-8") as f:
        categories = [line.strip() for line in f.readlines() if line.strip()]
    
    # Falls Teachable Machine Nummern vorangestellt hat (z. B. "0 Katze"), diese entfernen
    clean_categories = []
    for cat in categories:
        parts = cat.split(" ", 1)
        if len(parts) > 1 and parts[0].isdigit():
            clean_categories.append(parts[1])
        else:
            clean_categories.append(cat)
    return clean_categories

# ---------------------------------------------------------
# App Interface
# ---------------------------------------------------------
st.markdown("<h1 class='title-text'>🧠 KI-Bilderkennung</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Lade ein Bild hoch, um es von dem trainierten KI-Modell analysieren zu lassen.</p>", unsafe_allow_html=True)

# Modell und Labels laden
try:
    model = load_keras_model()
    categories = load_labels()
except Exception as e:
    st.error(f"Fehler beim Laden der Modell- oder Label-Datei: {e}")
    st.info("Bitte stelle sicher, dass 'keras_model.h5' und 'labels.txt' im selben Ordner liegen.")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader(
    "Wähle ein Foto aus...", 
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Bild öffnen & anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)
    
    with st.spinner("Analyse läuft..."):
        # Bild-Vorverarbeitung für MobileNet/Teachable Machine (224x224)
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image_resized, dtype=np.float32)
        
        # Normalisierung (-1 bis 1, Standard für Keras/Teachable Machine)
        normalized_image_array = (img_array / 127.5) - 1.0
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Vorhersage generieren
        predictions = model.predict(data)[0]
    
    st.divider()
    st.subheader("📊 Erkennungsergebnisse")
    
    # Sortieren nach Wahrscheinlichkeit (höchste zuerst)
    results = sorted(
        zip(categories, predictions), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Höchstes Ergebnis hervorheben
    top_category, top_score = results[0]
    top_percentage = top_score * 100
    
    st.success(f"**Hauptvorhersage:** {top_category} ({top_percentage:.1f}%)")
    
    st.write("---")
    st.write("**Alle Kategorien:**")
    
    # Liste aller Kategorien mit Prozentzahlen & Fortschrittsbalken
    for category, score in results:
        percentage = float(score * 100)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{category}**")
            st.caption(f"{percentage:.2f}%")
        with col2:
            st.progress(min(max(score, 0.0), 1.0))
