"""Evaluación de los avisos FRENAR_YA en todos los vídeos (Fase 8).

- Las detecciones de YOLO+ByteTrack se guardan en caché (outputs/cache/) la primera vez:
  los arreglos tocan risk/state/decider, no YOLO, así que todas las versiones ven las
  MISMAS cajas y la evaluación tarda segundos en vez de minutos.
- Se llama a la API cada EVERY_S segundos de vídeo, sin hilo: aquí importa QUÉ decide,
  no cuánto tarda, y así cada versión decide sobre los mismos instantes.

Uso (desde la raíz): python scripts/eval_alerts.py [etiqueta]
"""

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import openai

sys.path.insert(0, "src")
from decider import decide, interpret  # noqa: E402
from risk import SceneMemory  # noqa: E402
from state import build_state  # noqa: E402

EVERY_S = 0.5
# Un aviso es ACIERTO si empieza en esta ventana alrededor del choque (Nexar time_of_event).
HIT_WINDOW_BEFORE_S = 4.0
HIT_WINDOW_AFTER_S = 0.5
# Dos decisiones FRENAR separadas por menos de esto son el mismo episodio (el mismo aviso).
MERGE_GAP_S = 1.0
CACHE_DIR = Path("outputs/cache")

# Misma calibración que src/capture.py.
VIDEOS = {
    "positive/00021": (300, 530),
    "positive/00015": (450, 580),
    "positive/00000": (410, 545),
    "negative/01050": (395, 560),
    "negative/01074": (400, 560),
    "negative/01046": (590, 650),
}


def detections(name):
    """Lista por frame de [(id, clase, caja)], desde la caché o pasando YOLO una vez."""
    path = CACHE_DIR / f"{name.replace('/', '_')}.json"
    if path.exists():
        return json.loads(path.read_text())
    from ultralytics import YOLO  # solo si hace falta: cargar YOLO tarda

    model = YOLO("yolo26s.pt")
    cap = cv2.VideoCapture(f"data/videos/train/{name}.mp4")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(frame, device="mps", classes=[0, 1, 2, 3, 5, 7, 9, 11], persist=True,
                        tracker="bytetrack.yaml", verbose=False)[0]
        dets = []
        if r.boxes.id is not None:
            for xyxy, tid, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist()):
                dets.append((int(tid), r.names[int(c)], [round(v, 1) for v in xyxy]))
        frames.append(dets)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frames))
    return frames


def decide_with_retry(state, attempts=4):
    """En la evaluación no hay prisa: si la API falla de forma transitoria (timeout, red),
    reintentamos en vez de abortar. En el copiloto esto lo hace DecisionWorker."""
    for attempt in range(attempts):
        try:
            return decide(state)
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def run_video(name, horizon_y, hood_y):
    """Devuelve [(t, estado, probs, resumen)] cada EVERY_S segundos de vídeo."""
    fps = cv2.VideoCapture(f"data/videos/train/{name}.mp4").get(cv2.CAP_PROP_FPS)
    memory = SceneMemory(1280, horizon_y, hood_y)
    every = round(EVERY_S * fps)
    out, history = [], []
    for idx, dets in enumerate(detections(name), start=1):
        t = idx / fps
        memory.update(t, [(tid, cls, tuple(b)) for tid, cls, b in dets])
        if idx % every == 0:
            # Sin semáforo: TrafficLightMonitor necesita los píxeles y aquí solo tenemos cajas.
            state = build_state(memory, t, traffic_light=None, decision_history=history[-3:])
            probs = decide_with_retry(state)
            _, summary = interpret(probs)
            history.append(summary)
            out.append((t, state, probs, summary))
    return out


def episodes(decisions):
    """Agrupa las decisiones con FRENAR_YA en episodios: [(inicio, fin, decisión inicial)]."""
    eps = []
    for t, state, probs, summary in decisions:
        if "FRENAR_YA" not in summary.split(" + "):
            continue
        if eps and t - eps[-1][1] <= MERGE_GAP_S:
            eps[-1][1] = t
        else:
            eps.append([t, t, (state, probs)])
    return eps


def culprit(state):
    """El objeto con menor TTC del estado al empezar el aviso (el sospechoso)."""
    objs = state["objetos"]
    if not objs:
        return "sin objetos"
    o = objs[0]  # build_state los ordena por peligro
    return (f"{o['clase']} {o['zona']} d={o['distancia_rel']} lateral={o['lateral']} "
            f"ttc={o['ttc_s']} ttc_tend={o['ttc_tendencia_s']}")


label = sys.argv[1] if len(sys.argv) > 1 else "sin etiqueta"
with open("data/videos/train/positive/metadata.csv") as f:
    events = {"positive/" + r["file_name"].removesuffix(".mp4"): float(r["time_of_event"]) for r in csv.DictReader(f)}

print(f"##### Evaluación: {label}")
total_false, total_minutes, hits = 0, 0.0, []
for name, (horizon_y, hood_y) in VIDEOS.items():
    decisions = run_video(name, horizon_y, hood_y)
    event_t = events.get(name)
    eps = episodes(decisions)
    false_eps, hit = [], None
    for start, end, (state, _) in eps:
        if event_t is not None and event_t - HIT_WINDOW_BEFORE_S <= start <= event_t + HIT_WINDOW_AFTER_S:
            hit = hit or start
        elif event_t is None or start < event_t - HIT_WINDOW_BEFORE_S:
            false_eps.append((start, culprit(state)))
        # Lo que empieza DESPUÉS del choque no cuenta: el coche ya ha chocado.
    # Minutos "evaluables": en los positivos, solo hasta la ventana del choque.
    minutes = ((event_t - HIT_WINDOW_BEFORE_S) if event_t else decisions[-1][0]) / 60
    total_false += len(false_eps)
    total_minutes += minutes
    if event_t is not None:
        hits.append(hit is not None)
    result = ("" if event_t is None else
              f"  choque {event_t:.1f}s → " + (f"AVISO a {hit:.1f}s ({event_t - hit:.1f}s antes)" if hit else "SIN AVISO ❌"))
    print(f"\n{name}{result}  |  falsas alarmas: {len(false_eps)}")
    for start, who in false_eps:
        print(f"    {start:5.1f}s  {who}")

print(f"\nRESUMEN [{label}]: choques avisados {sum(hits)}/{len(hits)} | "
      f"falsas alarmas {total_false} en {total_minutes:.1f} min = {total_false / total_minutes:.1f}/min")
