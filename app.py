import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "rescuevision.db"
MODEL_NAME = "yolo11n.pt"
PERSON_CLASS = "person"

st.set_page_config(page_title="RescueVision AI", page_icon="🚨", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 1.2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; color: #fff;
       background: linear-gradient(120deg,#111827,#7f1d1d 72%,#ea580c); margin-bottom: 1rem;}
.hero h1 {color:#fff; margin:0; font-size:2.3rem;}
.hero p {color:#fee2e2; margin:.4rem 0 0 0;}
div[data-testid="stMetric"] {border:1px solid rgba(150,150,150,.25); padding:12px; border-radius:14px;}
.small-note {font-size:.88rem; opacity:.8;}
</style>
<div class="hero">
<h1>🚨 RescueVision AI</h1>
<p>Computer-vision-assisted incident triage · Human-in-the-loop review · Incident analytics</p>
</div>
""", unsafe_allow_html=True)

def connect_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            location TEXT,
            image_name TEXT,
            damage_level TEXT,
            people_detected INTEGER,
            object_count INTEGER,
            priority TEXT,
            notes TEXT,
            detections_json TEXT
        )
    """)
    con.commit()
    return con

def get_incidents():
    with connect_db() as con:
        return pd.read_sql_query("SELECT * FROM incidents ORDER BY id DESC", con)

def save_incident(record):
    with connect_db() as con:
        con.execute("""
            INSERT INTO incidents
            (created_at, location, image_name, damage_level, people_detected,
             object_count, priority, notes, detections_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["created_at"], record["location"], record["image_name"],
            record["damage_level"], record["people_detected"], record["object_count"],
            record["priority"], record["notes"], json.dumps(record["detections"])
        ))
        con.commit()

@st.cache_resource(show_spinner="Loading computer-vision model…")
def load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_NAME)

def decide_priority(damage_level, people_count):
    # Transparent demo rule, not a validated emergency triage protocol.
    if damage_level == "Severe" or people_count >= 3:
        return "HIGH"
    if damage_level == "Moderate" or people_count >= 1:
        return "MEDIUM"
    return "LOW"

with st.sidebar:
    st.header("Incident details")
    location = st.text_input("Location / incident ID", placeholder="e.g., Zone A / Incident 001")
    damage_level = st.select_slider("Human-reviewed visible damage", options=["Unknown", "Minor", "Moderate", "Severe"], value="Unknown")
    notes = st.text_area("Responder notes", placeholder="Visible hazards, access constraints, follow-up needed…")
    st.divider()
    st.caption("Priority uses a transparent exhibition rule. It is not an official emergency triage protocol.")
    conf = st.slider("Object detection confidence", 0.10, 0.90, 0.35, 0.05)

tab_analyze, tab_log, tab_method = st.tabs(["🖼️ Analyze scene", "🗂️ Incident log", "🧠 ML & methodology"])

with tab_analyze:
    st.subheader("Scene analysis")
    st.write("Upload a scene image. The pretrained model detects its supported everyday object classes; a human reviewer records the visible damage level.")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1, 1.1], gap="large")
        with left:
            st.image(image, caption="Uploaded scene", use_container_width=True)
        with right:
            st.markdown("#### Computer-vision results")
            if st.button("🔎 Analyze image", type="primary", use_container_width=True):
                try:
                    model = load_model()
                    results = model.predict(image, conf=conf, verbose=False)
                    result = results[0]
                    detections = []
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0].item())
                            detections.append({
                                "class": result.names[cls_id],
                                "confidence": round(float(box.conf[0].item()), 3),
                                "box_xyxy": [round(float(x), 1) for x in box.xyxy[0].tolist()]
                            })
                    annotated = Image.fromarray(result.plot()[:, :, ::-1])
                    people_count = sum(d["class"] == PERSON_CLASS for d in detections)
                    object_count = len(detections)
                    priority = decide_priority(damage_level, people_count)
                    st.session_state["analysis"] = {
                        "image_name": uploaded.name,
                        "annotated": annotated,
                        "detections": detections,
                        "people_count": people_count,
                        "object_count": object_count,
                        "priority": priority,
                        "damage_level": damage_level,
                        "location": location,
                        "notes": notes,
                    }
                except Exception as exc:
                    st.error("The detector could not load or run. Check your internet connection on the first run (the model weights may need to download), then retry.")
                    st.code(f"{type(exc).__name__}: {exc}")

            analysis = st.session_state.get("analysis")
            if analysis and analysis["image_name"] == uploaded.name:
                st.image(analysis["annotated"], caption="Model annotations", use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Objects detected", analysis["object_count"])
                c2.metric("People detected", analysis["people_count"])
                c3.metric("Review priority", analysis["priority"])
                st.caption("Detections are limited to classes the generic pretrained model knows. They do not establish structural damage or confirm casualties.")
                if analysis["detections"]:
                    det_df = pd.DataFrame([
                        {"Class": d["class"], "Confidence": d["confidence"], "Bounding box (pixels)": str(d["box_xyxy"])}
                        for d in analysis["detections"]
                    ])
                    st.dataframe(det_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No objects passed the selected confidence threshold.")

                if st.button("💾 Save incident to log", type="primary", use_container_width=True):
                    record = {
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "location": analysis["location"],
                        "image_name": analysis["image_name"],
                        "damage_level": analysis["damage_level"],
                        "people_detected": analysis["people_count"],
                        "object_count": analysis["object_count"],
                        "priority": analysis["priority"],
                        "notes": analysis["notes"],
                        "detections": analysis["detections"],
                    }
                    save_incident(record)
                    st.success("Incident saved to the local SQLite log.")
    else:
        st.info("Upload a JPG or PNG image to begin. For a reliable demo, prepare 2–3 sample images in advance.")

with tab_log:
    st.subheader("Incident log & analytics")
    incidents = get_incidents()
    if incidents.empty:
        st.info("No saved incidents yet. Analyze an image and save the incident to populate this page.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Saved incidents", len(incidents))
        b.metric("High priority", int((incidents["priority"] == "HIGH").sum()))
        c.metric("People detected (model)", int(incidents["people_detected"].sum()))
        d.metric("Locations entered", int(incidents["location"].fillna("").replace("", pd.NA).nunique()))
        chart = incidents["priority"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
        st.markdown("#### Review-priority distribution")
        st.bar_chart(chart)
        st.markdown("#### Latest incidents")
        visible_cols = ["id", "created_at", "location", "image_name", "damage_level", "people_detected", "object_count", "priority", "notes"]
        st.dataframe(incidents[visible_cols], use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export incident log (CSV)", incidents.drop(columns=["detections_json"]).to_csv(index=False).encode("utf-8"), "rescuevision_incidents.csv", "text/csv")

with tab_method:
    st.subheader("What the ML does — and what it doesn't")
    st.markdown("""
    **Pipeline**
    1. **Input:** an uploaded RGB image.
    2. **Preprocessing:** the image is passed to a pretrained YOLO object detector.
    3. **Inference:** the model returns supported object classes, confidence scores, and bounding boxes.
    4. **Human-in-the-loop:** the reviewer selects the visible damage level and enters incident notes.
    5. **Decision rules:** a transparent rule combines the reviewer-selected damage level with the number of detected people to suggest a review priority.
    6. **Persistence:** the incident record and detections are stored in a local SQLite database; the log can be exported to CSV.

    **Important limits**
    - This starter uses a general-purpose pretrained detector, not a model trained to recognize disaster damage.
    - A detected person is not proof that someone is trapped or injured. A missed detection is not proof that nobody is present.
    - The priority rule is an exhibition heuristic, not a validated triage standard.
    - Use only sample/test images for the exhibition. Never use this prototype to make real emergency decisions.
    """)
    st.code("""# Core inference idea
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # pretrained general-purpose detector
result = model.predict(image, conf=0.35, verbose=False)[0]
for box in result.boxes:
    class_name = result.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    bbox_xyxy = box.xyxy[0].tolist()
""", language="python")

st.caption("RescueVision AI · Educational prototype · Local SQLite storage · Human-reviewed incident triage")
