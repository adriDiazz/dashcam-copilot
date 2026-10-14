"""Evalúa las alarmas de TTC en varios vídeos, sin ventana.

Una "alarma" = objeto maduro, en zona centro, con TTC < umbral. Agrupamos los frames
seguidos en "episodios" (lo que el conductor percibiría como un aviso) y, en los vídeos
con choque, comparamos el primer aviso con el time_of_alert/time_of_event de Nexar.

Uso (desde la raíz del proyecto): python scripts/eval_ttc.py
"""

import csv
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, "src")  # para importar risk.py igual que capture.py
from risk import SceneMemory  # noqa: E402

MODEL_PATH = "yolo26s.pt"
CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]
THRESHOLDS_S = [2.0, 3.0]
# Dos alarmas separadas por menos de esto cuentan como el mismo episodio (aviso).
MERGE_GAP_FRAMES = 15

# Calibración medida a ojo sobre frames de cada vídeo: (HORIZON_Y, HOOD_Y).
VIDEOS = {
    "negative/01046": (590, 650),  # cámara muy inclinada hacia arriba: distancia poco fiable
    "negative/01074": (400, 560),
    "negative/01050": (395, 560),
    "positive/00021": (300, 530),
    "positive/00015": (450, 580),
}
OUT_DIR = Path("outputs/eval_ttc")


def load_events():
    """{"positive/00021": (time_of_alert, time_of_event)} desde el metadata.csv de Nexar."""
    events = {}
    with open("data/videos/train/positive/metadata.csv") as f:
        for row in csv.DictReader(f):
            name = "positive/" + row["file_name"].removesuffix(".mp4")
            events[name] = (float(row["time_of_alert"]), float(row["time_of_event"]))
    return events


def episodes(frames):
    """Agrupa una lista ordenada de frames en tramos [inicio, fin]."""
    runs = []
    for f in frames:
        if runs and f - runs[-1][1] <= MERGE_GAP_FRAMES:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return runs


def evaluate(model, name, horizon_y, hood_y):
    cap = cv2.VideoCapture(f"data/videos/train/{name}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    memory = SceneMemory(width, horizon_y, hood_y)
    alarms = {th: {} for th in THRESHOLDS_S}  # {umbral: {frame: [(id, clase, ttc)]}}
    frames = {}  # guardamos el frame anotado al inicio de cada alarma para mirarlo luego
    model.predictor = None  # tracker nuevo para cada vídeo: que no herede IDs del anterior
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        r = model.track(frame, device="mps", classes=CLASSES, persist=True,
                        tracker="bytetrack.yaml", verbose=False)[0]
        t = idx / fps
        dets = []
        if r.boxes.id is not None:
            for xyxy, tid, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist()):
                dets.append((int(tid), r.names[int(c)], xyxy))
        memory.update(t, dets)
        for tid, _, _ in dets:
            # risk.py puede descartar detecciones (p. ej. nuestro propio capó): no están en la memoria.
            if tid not in memory.history or memory.history[tid][-1].t != t:
                continue
            if not memory.is_mature(tid) or memory.zone(tid) != "centro":
                continue
            ttc = memory.ttc(tid)
            for th in THRESHOLDS_S:
                if ttc is not None and ttc < th:
                    alarms[th].setdefault(idx, []).append((tid, memory.history[tid][-1].cls, round(ttc, 1)))
                    if th == max(THRESHOLDS_S) and idx - 1 not in alarms[th]:
                        frames[idx] = r.plot()
    cap.release()
    return fps, idx, alarms, frames


model = YOLO(MODEL_PATH)
events = load_events()
OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, (horizon_y, hood_y) in VIDEOS.items():
    fps, total, alarms, frames = evaluate(model, name, horizon_y, hood_y)
    print(f"\n=== {name} ({total} frames, {total / fps:.0f} s)")
    if name in events:
        alert_t, event_t = events[name]
        print(f"    Nexar: alerta {alert_t:.1f} s, choque {event_t:.1f} s")
    for th in THRESHOLDS_S:
        runs = episodes(sorted(alarms[th]))
        desc = ", ".join(f"{a / fps:.1f}-{b / fps:.1f}s {alarms[th][a][0][1]}#{alarms[th][a][0][0]}"
                         for a, b in runs)
        print(f"    TTC<{th:.0f}s: {len(runs)} avisos ({len(alarms[th])} frames)  {desc}")
    for idx, img in frames.items():
        cv2.imwrite(str(OUT_DIR / f"{name.replace('/', '_')}_{idx}.jpg"), cv2.resize(img, (640, 360)))
