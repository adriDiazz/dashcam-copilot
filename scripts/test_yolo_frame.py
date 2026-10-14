"""Prueba de YOLO sobre UN solo frame, para ver qué devuelve antes de meterlo en el bucle.

Uso (desde la raíz del proyecto): python scripts/test_yolo_frame.py
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/videos/train/positive/00021.mp4"  # ciudad con tráfico delante
FRAME_INDEX = 300  # ~10 s a 30 FPS: un frame "del medio", con escena ya en marcha
OUTPUT_PATH = Path("outputs/deteccion.jpg")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Error: no se pudo abrir el vídeo {VIDEO_PATH}")

# Saltamos directamente al frame que queremos en vez de leer los 300 anteriores.
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ret, frame = cap.read()
cap.release()
if not ret:
    raise SystemExit(f"Error: no se pudo leer el frame {FRAME_INDEX}")

# La primera vez descarga los pesos (unos pocos MB) a la carpeta actual.
model = YOLO("yolo26n.pt")

# model(...) acepta varias imágenes a la vez, por eso devuelve una LISTA de resultados.
# verbose=False para que no imprima su propio resumen y veamos solo el nuestro.
results = model(frame, device="mps", verbose=False)
r = results[0]

print(f"Detecciones: {len(r.boxes)}")
# Los datos vienen como tensores de PyTorch (posiblemente en la GPU):
# .tolist() los pasa a listas normales de Python para poder imprimirlos.
for xyxy, cls_id, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
    name = r.names[int(cls_id)]  # cls viene como float (2.0) → int para buscar en el diccionario
    box = [round(v) for v in xyxy]  # píxeles con decimales → enteros, más legible
    print(f"{name:<14} {conf:.2f}  {box}")

# r.plot() devuelve una copia del frame (BGR) con las cajas ya dibujadas.
OUTPUT_PATH.parent.mkdir(exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), r.plot())
print(f"Imagen guardada en {OUTPUT_PATH}")
