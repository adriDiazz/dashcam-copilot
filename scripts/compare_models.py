"""Compara varios modelos YOLO sobre el mismo vídeo: velocidad y qué detectan.

Uso (desde la raíz del proyecto): python scripts/compare_models.py
La primera vez descarga los pesos que falten.
"""

import statistics
import time
from collections import Counter

import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/videos/train/positive/00021.mp4"
CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]  # los mismos que en src/capture.py
CONF_THRESHOLD = 0.25
WARMUP_FRAMES = 10  # los primeros frames son lentos (la GPU "calienta"): no cuentan

# (modelo, dispositivo): incluimos 26n en CPU para ver cuánto aporta la GPU.
RUNS = [
    ("yolo26n.pt", "mps"),
    ("yolo26s.pt", "mps"),
    ("yolo11n.pt", "mps"),
    ("yolo26n.pt", "cpu"),
]


def evaluate(model_path, device):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(VIDEO_PATH)
    times, counts, confs = [], Counter(), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = model(frame, device=device, classes=CLASSES, conf=CONF_THRESHOLD, verbose=False)[0]
        times.append(time.perf_counter() - t0)
        for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            counts[r.names[int(cls_id)]] += 1
            confs.append(conf)
    cap.release()
    # Mediana y no media: un pico suelto del sistema no debe deformar el resultado.
    ms = 1000 * statistics.median(times[WARMUP_FRAMES:])
    return ms, counts, statistics.mean(confs)


results = {f"{m} ({d})": evaluate(m, d) for m, d in RUNS}

names = ["car", "person", "truck", "traffic light", "bicycle", "bus", "motorcycle"]
print(f"{'modelo':<20}{'ms/frame':>9}{'conf media':>11}" + "".join(f"{n[:9]:>10}" for n in names))
for label, (ms, counts, conf) in results.items():
    print(f"{label:<20}{ms:>9.1f}{conf:>11.2f}" + "".join(f"{counts[n]:>10}" for n in names))
