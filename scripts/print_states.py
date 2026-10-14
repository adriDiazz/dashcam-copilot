"""Imprime la secuencia de estados JSON alrededor del choque de varios vídeos.

Sirve para comprobar si el estado "cuenta la historia": alguien que SOLO lea estas
líneas, sin ver el vídeo, ¿podría decidir si hay que avisar?

Uso (desde la raíz del proyecto): python scripts/print_states.py
"""

import csv
import json
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, "src")
from risk import SceneMemory  # noqa: E402
from state import build_state  # noqa: E402
from traffic_light import TrafficLightMonitor  # noqa: E402

MODEL_PATH = "yolo26s.pt"
CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]
EVERY_S = 0.5  # un estado cada medio segundo (más o menos la frecuencia a la que llamaremos a la API)
BEFORE_S, AFTER_S = 4.0, 0.5  # ventana alrededor del choque

# (HORIZON_Y, HOOD_Y) medidos a ojo para cada vídeo.
VIDEOS = {
    "positive/00000": (410, 545),  # noche: furgoneta que se mete desde la izquierda
    "positive/00021": (300, 530),  # día: el coche de delante frena de golpe
}

with open("data/videos/train/positive/metadata.csv") as f:
    events = {"positive/" + r["file_name"].removesuffix(".mp4"): r for r in csv.DictReader(f)}

model = YOLO(MODEL_PATH)
for name, (horizon_y, hood_y) in VIDEOS.items():
    alert_t, event_t = float(events[name]["time_of_alert"]), float(events[name]["time_of_event"])
    print(f"\n=== {name}  (alerta Nexar {alert_t:.1f} s, choque {event_t:.1f} s)")
    cap = cv2.VideoCapture(f"data/videos/train/{name}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    memory = SceneMemory(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), horizon_y, hood_y)
    lights = TrafficLightMonitor()
    model.predictor = None  # tracker nuevo por vídeo
    every_frames = round(EVERY_S * fps)
    idx = 0
    while True:
        ok, frame = cap.read()
        idx += 1
        t = idx / fps
        if not ok or t > event_t + AFTER_S:
            break
        r = model.track(frame, device="mps", classes=CLASSES, persist=True,
                        tracker="bytetrack.yaml", verbose=False)[0]
        dets = []
        if r.boxes.id is not None:
            for xyxy, tid, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist()):
                dets.append((int(tid), r.names[int(c)], xyxy))
        memory.update(t, dets)
        color = lights.update(frame, [b for _, cls, b in dets if cls == "traffic light"])
        if t >= event_t - BEFORE_S and idx % every_frames == 0:
            state = build_state(memory, t, traffic_light=color)
            mark = "  ⏰" if abs(t - alert_t) < EVERY_S / 2 else "  💥" if abs(t - event_t) < EVERY_S / 2 else ""
            # Una línea por estado y sin el historial (vacío hasta la Fase 6) para leerlo de un vistazo.
            state.pop("historial_decisiones")
            print(json.dumps(state, ensure_ascii=False) + mark)
    cap.release()
