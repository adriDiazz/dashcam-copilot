import sys
import time
from collections import deque

import cv2
from ultralytics import YOLO

# Funciona porque al ejecutar `python src/capture.py` la carpeta src/ entra en el path.
from risk import SceneMemory
import overlay
from decider import DecisionWorker, warm_up
from state import build_state
from traffic_light import TrafficLightMonitor

# Calibración de cada cámara (píxeles y): (HORIZON_Y = donde la carretera se pierde en el
# horizonte, HOOD_Y = donde empieza el capó), medida a ojo sobre un frame. Cada dashcam está
# a una altura y ángulo distintos, así que cada vídeo necesita la suya.
CALIBRATION = {
    "positive/00021": (300, 530),  # ciudad, el de delante frena de golpe
    "positive/00015": (450, 580),  # autopista, choque por alcance, contraluz
    "positive/00000": (410, 545),  # noche, furgoneta que se mete desde la izquierda
    "negative/01050": (395, 560),  # autopista con túnel, sin incidente
    "negative/01074": (400, 560),  # afueras con cuesta, sin incidente
    "negative/01046": (590, 650),  # cámara muy inclinada: distancias poco fiables
}
# Vídeo por argumento: `python src/capture.py positive/00015` (por defecto, 00021).
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "positive/00021"
if VIDEO not in CALIBRATION:
    raise SystemExit(f"No hay calibración para {VIDEO}. Opciones: {', '.join(CALIBRATION)}")
HORIZON_Y, HOOD_Y = CALIBRATION[VIDEO]
# Ruta relativa: hay que ejecutar el script desde la raíz del proyecto.
VIDEO_PATH = f"data/videos/train/{VIDEO}.mp4"
# Small: ~17 ms/frame en MPS (cabe en los 33 ms) y detecta ~3x más semáforos y +35% personas
# que nano (ver scripts/compare_models.py). Si nos pasamos de tiempo, volver a "yolo26n.pt".
MODEL_PATH = "yolo26s.pt"

# IDs de COCO que nos interesan; el resto (de 80 clases) no aporta nada en la carretera.
CLASSES = [
    0,  # person
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
    9,  # traffic light
    11,  # stop sign
]
# Sin conf= propio: con tracking, model.track() usa conf=0.1 porque ByteTrack necesita
# también las detecciones dudosas. Los umbrales reales están en bytetrack.yaml.
TRACKER = "bytetrack.yaml"

# Cuántos frames promediamos para el FPS. Más = número más estable pero tarda más
# en reaccionar a un cambio de velocidad. 30 frames ≈ medio segundo a 60 FPS.
FPS_WINDOW = 30

# Cargamos el modelo UNA vez fuera del bucle: cargarlo en cada frame sería lentísimo.
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

# VideoCapture NO lanza error si la ruta está mal: solo deja el vídeo "cerrado".
# Por eso hay que comprobarlo a mano y parar, o el bucle fallaría de forma confusa.
if not cap.isOpened():
    raise SystemExit(f"Error: no se pudo abrir el vídeo {VIDEO_PATH}")

# cv2.CAP_PROP_* son solo identificadores (3, 4, 5...); el valor real se pide con cap.get().
# get() siempre devuelve float, por eso convertimos a int lo que son números enteros.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)  # a qué velocidad se GRABÓ el vídeo
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Resolución: {width}x{height} | FPS originales: {video_fps:.1f} | Frames: {total_frames}")

first_frame = True
prev_time = time.perf_counter()

# deque con maxlen: al añadir el elemento 31, el más antiguo sale solo.
# Así siempre tenemos las duraciones de los últimos FPS_WINDOW frames sin gestionar nada.
frame_times = deque(maxlen=FPS_WINDOW)
# Tiempo de detección + tracking. El FPS total no basta: la pantalla lo limita a ~60.
yolo_times = deque(maxlen=FPS_WINDOW)

# Memoria de la escena (src/risk.py): historia de cada objeto por su ID de tracking.
memory = SceneMemory(frame_width=width, horizon_y=HORIZON_Y, hood_y=HOOD_Y)
# Color del semáforo que nos afecta, suavizado entre frames (src/traffic_light.py).
lights = TrafficLightMonitor()
# Cada cuántos frames construimos un estado y se lo dejamos al hilo de decisiones.
# Da igual si son más de los que la API puede atender: el hilo solo usa el más reciente.
STATE_EVERY_FRAMES = 5
# Una decisión sobre una escena de hace más de esto ya no describe lo que pasa: no se muestra.
MAX_DECISION_AGE_S = 1.5
# Reproducir a la velocidad a la que se grabó, como en un coche de verdad. Si no, el vídeo
# va a ~60 FPS y la API (que tarda en tiempo REAL) parece llegar el doble de tarde.
REAL_TIME = True

print("Calentando la conexión con la Decisions API...")
warm_up()  # la primera llamada tarda 1-3 s: mejor ahora que en la primera decisión real
decisions = DecisionWorker(min_interval_s=0.5)
beeper = overlay.Beeper()
debug = False  # tecla d
frame_idx = 0
start_wall = time.perf_counter()  # reloj real al empezar: para ir a la velocidad del vídeo

while True:
    ret, frame = cap.read()
    # ret es False cuando ya no quedan frames (fin del vídeo) o falla la lectura.
    if not ret:
        print("Fin del vídeo.")
        break

    if first_frame:
        # NumPy ordena como (filas, columnas, canales) = (alto, ancho, 3): al revés que "1280x720".
        print(f"frame.shape: {frame.shape}")
        first_frame = False

    frame_idx += 1

    # Cronometramos solo detección + tracking para saber cuánto cuestan de verdad.
    t0 = time.perf_counter()
    results = model.track(
        frame,
        device="mps",
        classes=CLASSES,  # filtra DENTRO de YOLO: ni devuelve las clases que no pedimos
        tracker=TRACKER,
        # persist=True: conserva la memoria del tracker entre llamadas.
        # Sin él, cada frame empezaría un tracker nuevo y los IDs no servirían de nada.
        persist=True,
        verbose=False,  # si no, Ultralytics imprime una línea por cada frame
    )
    yolo_times.append(time.perf_counter() - t0)
    r = results[0]

    # Tiempo del VÍDEO, no del reloj: lo reproducimos a ~60 FPS (más rápido de lo grabado)
    # y con el reloj real los coches parecerían acercarse al doble de velocidad.
    t_video = frame_idx / video_fps

    # boxes.id es None si en este frame el tracker no sigue ningún objeto.
    detections = []
    if r.boxes.id is not None:
        for xyxy, track_id, cls_id in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist()):
            detections.append((int(track_id), r.names[int(cls_id)], xyxy))
    memory.update(t_video, detections)
    # Semáforos: risk.py los ignora (no van por el suelo); su color se mira aparte.
    # r.orig_img es el frame SIN dibujar: si usáramos el de plot(), las cajas pintadas
    # contaminarían los colores.
    light_boxes = [xyxy for _, cls, xyxy in detections if cls == "traffic light"]
    light_color = lights.update(r.orig_img, light_boxes)

    # build_state() se ejecuta AQUÍ, en el hilo principal, porque lee SceneMemory, que este
    # bucle modifica en cada frame. Al hilo de decisiones solo le pasamos el diccionario
    # resultante (una "foto"), nunca la memoria compartida.
    if frame_idx % STATE_EVERY_FRAMES == 0:
        decisions.submit(build_state(memory, t_video, traffic_light=light_color))

    now = time.perf_counter()
    elapsed = now - prev_time  # cuánto ha durado esta vuelta
    prev_time = now  # guardar la hora para la próxima vuelta
    frame_times.append(elapsed)

    # Promediamos los TIEMPOS y luego invertimos, en vez de promediar FPS:
    # así un frame lento pesa lo que realmente ha durado.
    proc_fps = len(frame_times) / sum(frame_times)  # a qué velocidad PROCESAMOS nosotros
    yolo_ms = 1000 * sum(yolo_times) / len(yolo_times)

    # Dibujamos sobre el frame original (ya no usamos r.plot()): todo se pinta ANTES de imshow.
    # Última decisión de la API: llega con retraso ("edad" = vídeo transcurrido desde el
    # estado que se evaluó); si es muy vieja, draw_banner no la muestra.
    latest = decisions.latest()
    age_s = t_video - latest["state_t"] if latest is not None else 0.0
    overlay.draw_objects(frame, memory, t_video)
    if debug:
        overlay.draw_debug(frame, memory, light_color, latest)
    shown = overlay.draw_banner(frame, latest, age_s, MAX_DECISION_AGE_S)
    overlay.draw_hud(frame, proc_fps, yolo_ms, latest, age_s)
    beeper.update(shown)

    cv2.imshow("Copiloto dashcam", frame)

    # Sin waitKey la ventana no se refresca: es cuando OpenCV procesa los eventos de la ventana.
    # & 0xFF se queda con los 8 bits de la tecla (en algunos sistemas vienen bits extra).
    # Con REAL_TIME, waitKey además ESPERA lo que falte para que el frame N salga en el
    # instante N / fps desde el inicio. Si vamos con retraso, esperamos lo mínimo (1 ms).
    delay_ms = 1
    if REAL_TIME:
        ahead_s = start_wall + frame_idx / video_fps - time.perf_counter()
        delay_ms = max(1, int(ahead_s * 1000))
    key = cv2.waitKey(delay_ms) & 0xFF
    if key == ord("q"):
        break
    if key == ord("d"):  # alterna el modo depuración (trapecio, estelas, probabilidades)
        debug = not debug

# Liberar el vídeo y cerrar la ventana; en macOS, si no, la ventana se queda colgada.
cap.release()
cv2.destroyAllWindows()
