"""Color del semáforo por HSV: COCO solo dice "traffic light", no si está en rojo.

Idea: una luz ENCENDIDA es lo más saturado y brillante del semáforo. Contamos cuántos
píxeles muy vivos de cada tono hay dentro de la caja y gana el que más tenga.
"""

from collections import Counter, deque

import cv2
import numpy as np

# Rangos de tono (H) en OpenCV, que va de 0 a 179. El rojo está en los DOS extremos
# porque el tono es un círculo. El verde de los semáforos tira a turquesa, de ahí que llegue a 95.
HUE_RANGES = {
    "rojo": [(0, 10), (160, 179)],
    "ambar": [(11, 35)],
    "verde": [(40, 95)],
}
# Solo cuentan los píxeles muy vivos y brillantes (la luz encendida); las apagadas son oscuras.
MIN_SATURATION = 100
MIN_VALUE = 150
# Además, la luz encendida es lo MÁS brillante de su caja: exigimos estar cerca del máximo.
# Sin esto, la carcasa amarilla de los semáforos de EE. UU. (pintura viva al sol) salía "ambar".
RELATIVE_VALUE = 0.9
# Si el color ganador ocupa menos que esto de la caja, no nos fiamos: "desconocido".
MIN_LIT_FRACTION = 0.03
# Semáforos muy pequeños (lejanos) tienen tan pocos píxeles que el color es una lotería.
MIN_BOX_HEIGHT = 15
# Votación entre los últimos N frames para que el color no parpadee (≈ 0,3 s).
VOTE_WINDOW = 10


def classify_color(frame, box):
    """'rojo' / 'ambar' / 'verde' / 'desconocido' para la caja (x1, y1, x2, y2) de un semáforo."""
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]  # recorte: primero filas (y), luego columnas (x)
    if crop.size == 0:
        return "desconocido"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Percentil 99 y no el máximo: un solo píxel quemado (reflejo) no debe marcar el listón.
    bright = max(MIN_VALUE, RELATIVE_VALUE * np.percentile(v, 99))
    lit = (s >= MIN_SATURATION) & (v >= bright)  # máscara de píxeles "encendidos"

    counts = {}
    for color, ranges in HUE_RANGES.items():
        in_range = np.zeros_like(lit)
        for lo, hi in ranges:
            in_range |= (h >= lo) & (h <= hi)
        counts[color] = int(np.count_nonzero(lit & in_range))

    best = max(counts, key=counts.get)
    if counts[best] < MIN_LIT_FRACTION * crop.shape[0] * crop.shape[1]:
        return "desconocido"
    return best


class TrafficLightMonitor:
    """Elige el semáforo que nos afecta y suaviza su color en el tiempo."""

    def __init__(self):
        self.recent = deque(maxlen=VOTE_WINDOW)

    def update(self, frame, light_boxes):
        """light_boxes: cajas de los semáforos de este frame. Devuelve el color para el estado,
        o None si hace un rato que no vemos ninguno."""
        # Suponemos que el que nos afecta es el más GRANDE (el más cercano) que se pueda leer.
        # Es una simplificación: en un cruce puede haber varios para distintos carriles.
        readable = [b for b in light_boxes if b[3] - b[1] >= MIN_BOX_HEIGHT]
        if readable:
            main = max(readable, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            self.recent.append(classify_color(frame, main))
        else:
            self.recent.append(None)

        # Votación: el color más repetido en los últimos frames (None = no había semáforo).
        votes = Counter(c for c in self.recent if c is not None)
        if not votes or sum(votes.values()) < len(self.recent) / 2:
            return None
        return votes.most_common(1)[0][0]
