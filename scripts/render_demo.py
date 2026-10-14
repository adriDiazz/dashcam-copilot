"""Vídeo de demo para LinkedIn (1080x1080): dashcam con overlay arriba, panel de la API abajo.

Cada escena se procesa EN TIEMPO REAL (como en el coche), con el hilo de decisiones de
verdad: lo que se ve es lo que el sistema decidió en directo, con su latencia real.
Solo se graban los segundos de cada escena; antes se procesa el vídeo para "calentar"
el tracking y la memoria.

Uso (desde la raíz del proyecto):
  python scripts/render_demo.py                     → vídeo completo (outputs/demo_linkedin.mp4)
  python scripts/render_demo.py peatones semaforo   → solo esas escenas, de prueba (outputs/demo_test.mp4)
"""

import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

sys.path.insert(0, "src")
import overlay  # noqa: E402
from decider import WARN_THRESHOLD, DecisionWorker, warm_up  # noqa: E402
from risk import SceneMemory  # noqa: E402
from state import build_state  # noqa: E402
from traffic_light import TrafficLightMonitor  # noqa: E402

SIZE = 1080  # vídeo cuadrado: ocupa más pantalla en el móvil
VIDEO_H = 608  # 1280x720 escalado a 1080 de ancho
FPS = 30
RAW_PATH = "outputs/demo_raw.mp4"

# nombre: (vídeo, calibración, inicio, fin, título, choque real o None)
SCENE_LIBRARY = {
    "frenazo": ("positive/00021", (300, 530), 13.0, 22.0, "El coche de delante frena de golpe", 20.0),
    "autopista": ("positive/00015", (450, 580), 13.0, 21.5, "Alcance en autopista, a contraluz", 19.7),
    "adelantamientos": ("negative/01050", (395, 560), 16.0, 28.0, "Adelantamientos: sin falsas alarmas", None),
    "peatones": ("positive/00000", (410, 545), 0.5, 7.5, "Semáforo en rojo y peatones cruzando", None),
    "semaforo": ("negative/01040", (555, 650), 8.0, 28.0, "El semáforo pasa de verde a rojo", None),
}
DEFAULT_SCENES = ["frenazo", "autopista", "peatones", "adelantamientos"]
# Con nombres de escena por argumento, es una prueba: otro fichero, sin portada ni cierre.
TEST_MODE = len(sys.argv) > 1
SCENES = [SCENE_LIBRARY[k] for k in (sys.argv[1:] or DEFAULT_SCENES)]
OUT_PATH = "outputs/demo_test.mp4" if TEST_MODE else "outputs/demo_linkedin.mp4"

BG = (18, 18, 22)  # fondo del panel (RGB para PIL)
TEXT = (235, 235, 235)
MUTED = (150, 150, 160)
RED, AMBER, GREEN = (230, 60, 60), (255, 170, 0), (90, 200, 110)
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
font = {k: ImageFont.truetype(FONT, s) for k, s in {"title": 34, "h": 24, "txt": 22, "small": 19}.items()}
font["bold"] = ImageFont.truetype(FONT, 30, index=1)
font["mono"] = ImageFont.truetype(MONO, 20)
font["big"] = ImageFont.truetype(FONT, 56, index=1)

PROB_LABELS = {
    "frenar_ya": "frenar_ya",
    "mantener_distancia": "mantener_distancia",
    "peaton_en_trayectoria": "peaton_en_trayectoria",
    "semaforo_atencion": "semaforo_atencion",
}


def card(lines, seconds):
    """Fotogramas de una tarjeta de texto (intro/cierre). lines: [(texto, fuente, color)]."""
    img = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)
    total = sum(font[f].size + 22 for _, f, _ in lines)
    y = (SIZE - total) // 2
    for text, f, color in lines:
        w = d.textlength(text, font=font[f])
        d.text(((SIZE - w) / 2, y), text, font=font[f], fill=color)
        y += font[f].size + 22
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return [frame] * int(seconds * FPS)


def panel(title, scene_idx, decision, age_s, t_video, event_t):
    """Panel inferior: título, estado enviado a la API y sus 4 probabilidades."""
    img = Image.new("RGB", (SIZE, SIZE - VIDEO_H), BG)
    d = ImageDraw.Draw(img)
    d.text((36, 22), f"{scene_idx}/{len(SCENES)} · {title}", font=font["bold"], fill=TEXT)
    clock = f"t = {t_video:4.1f} s" + (f"   ·   choque real: {event_t:.1f} s" if event_t else "   ·   sin incidente")
    d.text((36, 64), clock, font=font["small"], fill=MUTED)

    # Izquierda: lo que se ENVÍA (el objeto más peligroso del estado).
    d.text((36, 108), "Estado enviado (JSON)", font=font["h"], fill=MUTED)
    fresh = decision is not None and decision.get("probs") and age_s <= 1.5
    obj = decision["state"]["objetos"][0] if fresh and decision["state"]["objetos"] else None
    light = json_value(decision["state"]["semaforo"]) if fresh else "null"
    if obj:
        keys = ["clase", "zona", "distancia_rel", "acercandose", "ttc_tendencia_s", "lateral"]
        lines = ['{"objetos": [{'] + [f'    "{k}": {json_value(obj[k])},' for k in keys] + ["  }],"]
    else:
        lines = ['{"objetos": [],']
    lines.append(f'  "semaforo": {light}}}')
    for k, line in enumerate(lines):
        d.text((36, 140 + 26 * k), line, font=font["mono"], fill=TEXT)

    # Derecha: lo que DEVUELVE la Decisions API.
    x0 = 560
    d.text((x0, 108), "Decisions API · gpt-6-luna", font=font["h"], fill=MUTED)
    for k, (name, label) in enumerate(PROB_LABELS.items()):
        y = 150 + 62 * k
        p = decision["probs"].get(name) if fresh else None
        d.text((x0, y), label, font=font["small"], fill=TEXT)
        bx1, bx2, by = x0, SIZE - 110, y + 28
        d.rectangle([bx1, by, bx2, by + 14], outline=MUTED)
        if p is not None:
            color = RED if p >= WARN_THRESHOLD else AMBER if p >= 0.4 else GREEN
            d.rectangle([bx1, by, bx1 + int((bx2 - bx1) * p), by + 14], fill=color)
            d.text((bx2 + 14, by - 6), f"{p:.2f}", font=font["txt"], fill=color)
        # Marca del umbral de aviso (0.7): a partir de ahí, la barra se pone roja y se avisa.
        tx = bx1 + int((bx2 - bx1) * WARN_THRESHOLD)
        d.line([tx, by - 4, tx, by + 18], fill=TEXT, width=2)
    if fresh:
        d.text((x0, 410), f"latencia {decision['latency_ms']:.0f} ms", font=font["small"], fill=MUTED)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def json_value(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return f'"{v}"' if isinstance(v, str) else str(v)


def render_scene(model, name, calib, start_s, end_s, title, event_t, scene_idx, writer):
    cap = cv2.VideoCapture(f"data/videos/train/{name}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    memory = SceneMemory(1280, *calib)
    lights = TrafficLightMonitor()
    decisions = DecisionWorker(min_interval_s=0.5)
    model.predictor = None  # tracker nuevo para cada vídeo
    start_wall, idx, last_shown = time.perf_counter(), 0, "-"
    frames_shown, frames_gap = 0, 0  # para medir los huecos "sin decisión" de la escena
    while True:
        ok, frame = cap.read()
        idx += 1
        t = idx / fps
        if not ok or t > end_s:
            break
        r = model.track(frame, device="mps", classes=[0, 1, 2, 3, 5, 7, 9, 11], persist=True,
                        tracker="bytetrack.yaml", verbose=False)[0]
        dets = []
        if r.boxes.id is not None:
            for xyxy, tid, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist()):
                dets.append((int(tid), r.names[int(c)], xyxy))
        memory.update(t, dets)
        color = lights.update(r.orig_img, [b for _, cls, b in dets if cls == "traffic light"])
        # Solo pedimos decisiones cerca de la escena (ahorra llamadas), con 2 s de margen.
        if idx % 5 == 0 and t >= start_s - 2.0:
            decisions.submit(build_state(memory, t, traffic_light=color))

        if t >= start_s:
            latest = decisions.latest()
            age_s = t - latest["state_t"] if latest is not None else 0.0
            overlay.draw_objects(frame, memory, t)
            shown = overlay.draw_banner(frame, latest, age_s, 1.5)
            # Hueco = sin decisión reciente (API lenta o con errores). El primer segundo no
            # cuenta: es normal que la primera decisión tarde un poco en llegar.
            if t >= start_s + 1.0:
                frames_shown += 1
                frames_gap += shown is None
            if shown != last_shown:  # registro de cambios del aviso, para revisar sin ver el vídeo
                print(f"    {t:5.1f} s  {shown}  (semáforo: {color})")
                last_shown = shown
            top = cv2.resize(frame, (SIZE, VIDEO_H), interpolation=cv2.INTER_AREA)
            writer.write(np.vstack([top, panel(title, scene_idx, latest, age_s, t, event_t)]))

        # Tiempo real: el hilo de la API trabaja con su latencia de verdad respecto al vídeo.
        ahead = start_wall + idx / fps - time.perf_counter()
        if ahead > 0:
            time.sleep(ahead)
    cap.release()
    return frames_gap / max(frames_shown, 1)


writer = cv2.VideoWriter(RAW_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
for f in [] if TEST_MODE else card([("Copiloto con dashcam", "big", TEXT),
               ("YOLO ve · OpenAI Decisions API decide", "title", MUTED),
               ("4 preguntas tipadas con probabilidades, ~300 ms", "txt", MUTED)], 3.0):
    writer.write(f)

warm_up()
model = YOLO("yolo26s.pt")
gaps = []
for i, (name, calib, start_s, end_s, title, event_t) in enumerate(SCENES, start=1):
    print(f"Escena {i}: {name} ({start_s}-{end_s} s)...")
    gaps.append(render_scene(model, name, calib, start_s, end_s, title, event_t, i, writer))

# Solo afirmaciones que el propio vídeo (o la evaluación, con su tamaño) respaldan.
for f in [] if TEST_MODE else card([("Avisó de mantener la distancia segundos antes de cada choque", "bold", TEXT),
               ("Falsas alarmas: de 8,9 a 2,8 por minuto (evaluado en 6 vídeos)", "txt", TEXT),
               ("Vídeos: Nexar Collision Prediction Dataset (Nexar Open Data License)", "small", MUTED),
               ("Proyecto educativo · no es un sistema de seguridad", "small", MUTED)], 4.0):
    writer.write(f)
writer.release()

# mp4v no se reproduce bien en todas partes: lo pasamos a H.264 (yuv420p) para LinkedIn.
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", RAW_PATH, "-c:v", "libx264", "-crf", "20",
                "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT_PATH], check=True)
print(f"Listo: {OUT_PATH}")
# Línea que lee scripts/render_when_ready.py para aceptar o repetir el vídeo.
print("HUECOS: " + " ".join(f"{g:.3f}" for g in gaps))
