"""Fundbox — Fundbüro fürs Katharineum zu Lübeck.

Foto rein, KI sagt was es ist, Fund landet im Katalog.
Modell: keras_model.h5 (Teachable Machine), Klassen: labels.txt.
UI: streamlit-shadcn-ui + native Streamlit-Widgets für Upload/Suche.
"""

from __future__ import annotations

import datetime
import io
import json
import uuid
from pathlib import Path

import streamlit as st
import streamlit_shadcn_ui as ui
from PIL import Image

from teachable_ai import heuristic_guess, load_teachable_labels, predict_teachable

# --------------------------------------------------------------------------
# Setup & Konstanten
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Fundbox · Katharineum",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
ITEMS_FILE = DATA_DIR / "items.json"
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

CLASS_EMOJI = {
    "Kleidungsstücke": "👕",
    "Schulsachen": "📚",
    "Trinkflasche": "🥤",
    "Brotdose": "🍱",
    "Regenschirm": "☂️",
    "Schlüssel": "🔑",
    "Kopfhörer": "🎧",
    "Powerbank/Ladekabel": "🔋",
    "Brille": "👓",
    "Geldtasche": "👛",
    "Taschenrechner": "🧮",
}

TABS = ["Stöbern", "Suchen", "Fund melden"]
STATUS_VARIANT = {"Offen": "destructive", "Abgeholt": "outline"}

# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
def load_items() -> list:
    if ITEMS_FILE.exists():
        try:
            return json.loads(ITEMS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_items(items: list) -> None:
    ITEMS_FILE.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if "items" not in st.session_state:
    st.session_state["items"] = load_items()
items: list = st.session_state["items"]

st.session_state.setdefault("tab", TABS[0])
st.session_state.setdefault("flash", None)
st.session_state.setdefault("rep_bytes", None)
st.session_state.setdefault("rep_ai", None)
st.session_state.setdefault("rep_name_ai", "")
st.session_state.setdefault("done_id", None)
st.session_state.setdefault("uploader_nonce", 0)

CLASSES = load_teachable_labels()
heute = datetime.date.today().isoformat()


def flash(kind: str, title: str, text: str = "") -> None:
    st.session_state["flash"] = (kind, title, text)


def show_flash() -> None:
    msg = st.session_state.get("flash")
    if msg:
        kind, title, text = msg
        ui.alert(title, text or None,
                 variant="destructive" if kind == "error" else "default",
                 key="flash_alert")
        st.session_state["flash"] = None


def goto(view: str) -> None:
    st.session_state["tab"] = view
    st.rerun()


def emoji_for(name: str) -> str:
    return CLASS_EMOJI.get(name, "❓")


def photo(item: dict):
    fn = item.get("filename") or ""
    p = IMAGE_DIR / fn
    if fn and p.is_file():
        st.image(str(p), use_container_width=True)
    else:
        st.markdown(
            f"<div style='font-size:3.2rem;text-align:center;padding:1rem;'>"
            f"{emoji_for(item.get('category', ''))}</div>",
            unsafe_allow_html=True,
        )


def search(items: list, query: str) -> list:
    words = query.lower().strip().split()
    if not words:
        return items
    hits = []
    for it in items:
        text = " ".join([
            it.get("name", ""), it.get("category", ""),
            it.get("description", ""), " ".join(it.get("search_terms", [])),
        ]).lower()
        score = sum(1 for w in words if w in text)
        if score:
            hits.append((score, it))
    hits.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in hits]


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background: #FAFAF9; }
      section[data-testid="stSidebar"] { display: none; }
      .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 880px !important; margin-left: auto !important;
        margin-right: auto !important; }
      h1.fund-title { font-weight: 800; letter-spacing: -.03em; margin: 0;
                      font-size: clamp(2rem, 5vw, 3rem); color: #1C1917; }
      p.fund-sub { color: #57534E; margin: .2rem 0 0; font-size: 1.02rem; }
      .fund-hero { background: #FFFFFF; color: #1C1917;
                   border: 1px solid #E7E5E4; border-left: 6px solid #B91C1C;
                   border-radius: 1.1rem; padding: 1.5rem 1.7rem;
                   box-shadow: 0 4px 16px rgba(28,25,23,.06); }
      .fund-hero h2 { margin: 0; letter-spacing: -.02em; color: #1C1917; }
      .fund-hero p { margin: .3rem 0 0; color: #57534E; }
      .school-seal { width: 92px; height: 92px; border-radius: 50%;
                     border: 2px solid #B91C1C; display: flex;
                     align-items: center; justify-content: center;
                     font-size: 2.2rem; margin: 0 auto; background: #FFFFFF;
                     color: #B91C1C; }
      .school-cap { text-align: center; color: #57534E; font-size: .72rem;
                    letter-spacing: .06em; margin-top: .4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

h1, h2 = st.columns([5, 1])
with h1:
    st.markdown('<h1 class="fund-title">Fundbox 🔎</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fund-sub">Das Fundbüro des Katharineums zu Lübeck — '
        "Foto hochladen, KI erkennt den Gegenstand, fertig.</p>",
        unsafe_allow_html=True,
    )
    ui.badges([
        ("Katharineum zu Lübeck", "default"),
        (f"{len(CLASSES)} erkennbare Kategorien", "secondary"),
        ("KI: keras_model.h5", "outline"),
    ], key="head_badges")
with h2:
    st.markdown('<div class="school-seal">🎒</div>', unsafe_allow_html=True)
    st.markdown('<div class="school-cap">KATHARINEUM<br>ZU LÜBECK</div>',
                unsafe_allow_html=True)
ui.separator()

tab = ui.tabs(TABS, value=st.session_state.get("tab", TABS[0]), key="main_tabs")
st.session_state["tab"] = tab
show_flash()

# --------------------------------------------------------------------------
# Tab: Stöbern
# --------------------------------------------------------------------------
if tab == "Stöbern":
    st.markdown(
        "<div class='fund-hero'><h2>Etwas verloren? 👀</h2>"
        "<p>Stöbere durch die Fundstücke oder lass dir per Foto helfen — "
        "die KI erkennt Kleidung, Flaschen, Schlüssel & Co. automatisch.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        ui.metric_card("Fundstücke gesamt", len(items), description="im Katalog",
                       key="m_total")
    with c2:
        ui.metric_card("Kategorien", len(CLASSES), description="erkennt die KI",
                       key="m_cats")
    with c3:
        week = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        ui.metric_card(
            "Neu diese Woche",
            len([i for i in items if str(i.get("created_at", "")) >= week]),
            description="frisch eingetroffen",
            key="m_week",
        )

    st.write("")
    if ui.button("📷 Fund melden", key="hero_go"):
        goto("Fund melden")

    st.write("")
    st.subheader("Kategorien")
    cols = st.columns(4)
    for idx, cls in enumerate(CLASSES):
        n = len([i for i in items if i.get("category") == cls])
        with cols[idx % 4]:
            ui.card(title=f"{emoji_for(cls)} {cls}", description=f"{n} Stück",
                      key=f"catcard_{idx}")
            if ui.button("Ansehen", key=f"cat_{idx}", variant="outline"):
                st.session_state["q"] = cls
                goto("Suchen")

    st.write("")
    st.subheader("Kürzlich gefunden")
    fresh = items[:8]
    if not fresh:
        ui.alert("Noch nichts da",
                 "Sobald etwas gefunden wird, erscheint es hier. Jetzt schon etwas "
                 "gefunden? Dann melde es!", key="empty_alert")
        if ui.button("Ersten Fund melden", key="empty_go", variant="secondary"):
            goto("Fund melden")
    else:
        cols = st.columns(4)
        for idx, it in enumerate(fresh):
            with cols[idx % 4]:
                photo(it)
                st.markdown(f"**{it.get('name', 'Fundstück')}**")
                st.caption(it.get("category", ""))
                ui.badge(it.get("category", ""), variant="secondary",
                         key=f"freshbadge_{idx}_{it.get('id')}")

# --------------------------------------------------------------------------
# Tab: Suchen
# --------------------------------------------------------------------------
elif tab == "Suchen":
    st.subheader("Suchen")
    q = ui.input("Suchbegriff", value=st.session_state.get("q", ""), key="q",
                 placeholder="z. B. Trinkflasche, Schlüssel, blau …", type="search")
    kat = ui.select("Kategorie filtern", ["Alle", *CLASSES], key="fkat")
    st.caption(f"**{len(items)}** Fundstücke im Katalog")

    hits = search(items, q or "")
    if kat != "Alle":
        hits = [h for h in hits if h.get("category") == kat]
    st.caption(f"**{len(hits)}** Treffer")

    if not hits:
        ui.alert("Nichts gefunden",
                 "Anderen Begriff versuchen — oder war es vielleicht noch gar "
                 "nicht im Fundbüro? Dann melde den Verlust im Sekretariat.",
                 key="nohit_alert")
    else:
        for row in range(0, len(hits), 3):
            cols = st.columns(3)
            for k, it in enumerate(hits[row:row + 3]):
                with cols[k]:
                    ui.card(
                        title=f"{emoji_for(it.get('category', ''))} {it.get('name', 'Fundstück')}",
                        description=it.get("category", ""),
                        content=it.get("description", ""),
                        key=f"hitcard_{it.get('id')}",
                    )
                    photo(it)
                    if it.get("created_at"):
                        st.caption(f"Gefunden am {it['created_at'][:10]}")
                    if ui.button("Abgeholt ✅", key=f"got_{it.get('id')}",
                                 variant="outline"):
                        items.remove(it)
                        save_items(items)
                        flash("ok", "Abgeholt! 🎉",
                              "Der Eintrag wurde aus dem Katalog entfernt.")
                        st.rerun()

    st.write("")
    st.subheader("Verteilung")
    import pandas as pd

    rows = [{"Kategorie": c,
             "Anzahl": len([i for i in items if i.get("category") == c])}
            for c in CLASSES if any(i.get("category") == c for i in items)]
    if rows:
        ui.bar_chart(pd.DataFrame(rows), x="Kategorie", y="Anzahl",
                     title="Fundstücke je Kategorie", key="chart_kat")
    else:
        ui.alert("Keine Daten", "Noch keine Fundstücke eingetragen.",
                 key="nodata_alert")

# --------------------------------------------------------------------------
# Tab: Fund melden
# --------------------------------------------------------------------------
elif tab == "Fund melden":
    st.subheader("Etwas gefunden? 📷")
    ui.card(
        title="In einer Minute eingetragen",
        description="Foto hochladen → KI erkennt den Gegenstand → prüfen & speichern.",
        key="howto_card",
    )

    src = st.radio("Quelle", ["Datei hochladen", "Kamera"],
                   horizontal=True, key="rep_src")
    key = f"up_{st.session_state['uploader_nonce']}"
    up = (st.file_uploader("Foto", type=["jpg", "jpeg", "png", "webp"],
                           label_visibility="collapsed", key=key)
          if src == "Datei hochladen"
          else st.camera_input("Kamera", label_visibility="collapsed", key=key))
    if up is not None:
        st.session_state["rep_bytes"] = up.getvalue()

    if not st.session_state.get("rep_bytes"):
        ui.alert("Noch kein Foto",
                 "Lade ein Foto hoch oder nutze die Kamera — dann startet die KI.",
                 key="nophoto_alert")
        with st.expander("Fototipps 💡"):
            st.markdown(
                "- Gute Belichtung, ruhiger Hintergrund\n"
                "- Gegenstand vollständig und formatfüllend\n"
                "- Nicht zu weit weg, keine Personen im Bild"
            )
    else:
        pil = Image.open(io.BytesIO(st.session_state["rep_bytes"])).convert("RGB")
        p1, p2 = st.columns([2, 3])
        with p1:
            st.image(pil, caption="Vorschau", use_container_width=True)
        with p2:
            if ui.button("🔎 Gegenstand erkennen", key="ai_go"):
                with st.spinner("KI analysiert das Foto …"):
                    ai = predict_teachable(pil) or heuristic_guess(pil)
                    st.session_state["rep_ai"] = ai
                ai = st.session_state["rep_ai"]
                st.session_state["rep_name_ai"] = ai["label"].capitalize()
                st.rerun()

            ai = st.session_state.get("rep_ai")
            if ai:
                ui.card(
                    title=f"Erkannt: {ai['label'].capitalize()} "
                          f"{emoji_for(ai['label'])}",
                    description=f"Kategorie „{ai['category']}“ · {ai['engine']}",
                    key="ai_card",
                )
                ui.progress(min(100.0, max(0.0, ai["confidence"] * 100.0)),
                            label="Sicherheit", show_value=True, key="ai_prog")
                if ai.get("top3"):
                    st.caption("Top-3 der KI:")
                    for j, (lab, prob) in enumerate(ai["top3"]):
                        st.write(f"{emoji_for(lab)} {lab.capitalize()} — "
                                 f"{prob * 100:.0f} %")
                        ui.progress(prob * 100.0, show_value=False,
                                    key=f"ai_top3_{j}")
                if ai["confidence"] < 0.5:
                    ui.alert("Unsicher",
                             "Bitte Name und Kategorie unten von Hand prüfen.",
                             key="ai_warn")

        st.write("")
        ui.separator()
        st.subheader("Eintragen")
        ai = st.session_state.get("rep_ai") or {}
        name = ui.input("Bezeichnung", value=st.session_state.get("rep_name_ai", ""),
                        key="rep_name", placeholder="z. B. Rote Trinkflasche")
        cat_default = ai.get("category")
        kat = ui.select(
            "Kategorie", CLASSES,
            index=CLASSES.index(cat_default) if cat_default in CLASSES else 0,
            key="rep_kat",
        )
        desc = ui.textarea("Beschreibung", key="rep_desc",
                           placeholder="Farbe, Marke, Besonderheiten …", rows=3)

        if ui.button("In die Fundbox legen ✅", key="rep_save"):
            if not (name or "").strip():
                flash("error", "Bezeichnung fehlt",
                      "Bitte einen Namen für den Gegenstand angeben.")
                st.rerun()
            item_id = str(uuid.uuid4())
            ext = ".jpg"
            filename = item_id + ext
            rgb = pil
            if rgb.mode in ("RGBA", "P"):
                rgb = rgb.convert("RGB")
            rgb.save(IMAGE_DIR / filename, format="JPEG", quality=85)
            label = (name or "").strip()
            items.insert(0, {
                "id": item_id,
                "name": label,
                "category": kat if kat in CLASSES else "Sonstiges",
                "description": (desc or "").strip()
                or f"Automatisch erkanntes Objekt ({label}).",
                "search_terms": [label.lower(), (kat or "").lower()],
                "filename": filename,
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            })
            save_items(items)
            st.session_state["rep_bytes"] = None
            st.session_state["rep_ai"] = None
            st.session_state["rep_name_ai"] = ""
            for k in ("rep_name", "rep_kat", "rep_desc"):
                st.session_state.pop(k, None)
            st.session_state["uploader_nonce"] += 1
            st.session_state["done_id"] = label
            st.session_state["show_done"] = True
            st.rerun()

    if st.session_state.get("show_done"):
        choice = ui.alert_dialog(
            True, "Eingetragen! 🎉",
            f"„{st.session_state.get('done_id') or 'Der Fund'}“ ist jetzt in der "
            "Fundbox. Wie geht's weiter?",
            confirm_label="Weiteren Fund melden",
            cancel_label="Stöbern",
            key="done_dialog",
        )
        if choice is True:
            st.session_state["show_done"] = False
            st.rerun()
        elif choice is False:
            st.session_state["show_done"] = False
            goto("Stöbern")

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.write("")
ui.separator()
st.caption("Fundbox · Katharineum zu Lübeck · Foto rein, KI erkennt, abholen 🎒")
