"""Overlay para el conductor: se tiene que entender en medio segundo.

Principios: el color es el mensaje, una sola cosa grande (el aviso más grave),
solo se dibuja lo que importa, y la depuración va aparte (tecla "d").
"""

import subprocess
import time

import cv2
import numpy as np

from state import CLASS_GROUPS, NEAR_DISTANCE, VULNERABLE

FONT = cv2.FONT_HERSHEY_SIMPLEX
# Colores en BGR (OpenCV), no RGB.
GRAY = (160, 160, 160)
GREEN = (80, 200, 80)
AMBER = (0, 170, 255)
ORANGE = (0, 110, 255)
YELLOW = (0, 220, 255)
RED = (40, 40, 230)
WHITE = (255, 255, 255)

# Riesgo de cada caja, calculado por NOSOTROS en cada frame (no esperamos a la API,
# que decide sobre la escena entera y llega ~0,6 s tarde).
DANGER_TTC_S = 2.5
ATTENTION_TTC_S = 5.0

# Aspecto del aviso grande según el resumen de la API, por orden de gravedad.
BANNERS = {
    "FRENAR_YA": ("FRENA YA", RED, "frenar_ya"),
    "PEATON_CERCA": ("PEATON CERCA", ORANGE, "peaton_en_trayectoria"),
    "SEMAFORO_AMBAR_ROJO": ("SEMAFORO", AMBER, "semaforo_atencion"),
    "MANTENER_DISTANCIA": ("MANTEN LA DISTANCIA", YELLOW, "mantener_distancia"),
}
# Solo suenan los avisos urgentes, y SOLO cuando empiezan (no mientras duran): con un
# pitido cada 2 s, tras el choque de 00021 sonaba 11 veces seguidas. Un aviso que no
# para acaba ignorado (o con el sistema apagado); el banner ya recuerda que sigue activo.
SOUNDS = {
    "FRENAR_YA": "/System/Library/Sounds/Sosumi.aiff",
    "PEATON_CERCA": "/System/Library/Sounds/Ping.aiff",
}
# Aun así, dos avisos que empiezan muy seguidos (parpadeo del umbral) no suenan dos veces.
SOUND_COOLDOWN_S = 2.0


def _object_style(memory, track_id):
    """(color, grosor, etiqueta) de la caja de un objeto, o None si no se dibuja."""
    obs = memory.history[track_id][-1]
    if not memory.is_mature(track_id) or obs.cls not in CLASS_GROUPS:
        return None
    group = CLASS_GROUPS[obs.cls]
    # Mismo criterio de relevancia que el estado (state.py): así lo que ves en gris
    # es justo lo que la API NO recibe.
    relevant = (memory.zone(track_id) == "centro" or memory.distance_rel(track_id) < NEAR_DISTANCE
                or group in VULNERABLE)
    if not relevant:
        return GRAY, 1, ""
    # Igual que en state.py: el TTC solo significa "choque" en nuestro carril. Si no, un coche
    # aparcado al que nos acercamos de lado salía en rojo aunque la API (bien) no avisara.
    ttcs = []
    if memory.zone(track_id) == "centro":
        ttcs = [v for v in (memory.ttc(track_id), memory.ttc_trend(track_id)) if v is not None]
    ttc = min(ttcs, default=None)
    if ttc is not None and ttc < DANGER_TTC_S:
        return RED, 4, f"{group} {ttc:.1f}s"
    if ttc is not None and ttc <= ATTENTION_TTC_S:
        return AMBER, 3, f"{group} {ttc:.1f}s"
    return GREEN, 2, group


def draw_objects(frame, memory, t):
    """Cajas coloreadas por riesgo, solo de lo visto en este instante."""
    for track_id, hist in memory.history.items():
        if hist[-1].t != t:
            continue
        style = _object_style(memory, track_id)
        if style is None:
            continue
        color, thickness, label = style
        obs = hist[-1]
        p1, p2 = (int(obs.x1), int(obs.y1)), (int(obs.x2), int(obs.y2))
        cv2.rectangle(frame, p1, p2, color, thickness)
        if label:
            # Etiqueta corta DEBAJO de la caja: arriba se amontonaban con los semáforos.
            cv2.putText(frame, label, (p1[0], p2[1] + 22), FONT, 0.6, color, 2, cv2.LINE_AA)


def _panel(frame, x1, y1, x2, y2, color, alpha=0.75):
    """Rectángulo semitransparente: se lee el texto sin tapar del todo la carretera."""
    roi = frame[y1:y2, x1:x2]
    fill = np.full_like(roi, color)
    cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_banner(frame, decision, age_s, max_age_s):
    """El aviso grande + barra de confianza. Devuelve el resumen mostrado (para el sonido)."""
    width = frame.shape[1]
    if decision is None or decision["error"] or age_s > max_age_s:
        # Sin decisión fiable: indicador pequeño y gris. Mejor nada que un aviso viejo.
        text = "sin conexion" if decision is not None and decision["error"] else "sin decision"
        _small_indicator(frame, text, GRAY)
        return None

    summary = decision["summary"]
    shown = next((k for k in BANNERS if k in summary.split(" + ")), None)
    if shown is None:  # TODO_OK o INCIERTO: no merece un banner
        _small_indicator(frame, "OK" if summary == "TODO_OK" else "incierto", GREEN if summary == "TODO_OK" else GRAY)
        return summary

    text, color, prob_key = BANNERS[shown]
    # FRENA YA parpadea (4 veces por segundo): el movimiento atrae la mirada más que el color.
    if shown == "FRENAR_YA" and int(time.perf_counter() * 4) % 2:
        color = (20, 20, 120)
    x1, x2, y1, y2 = width // 2 - 260, width // 2 + 260, 20, 120
    _panel(frame, x1, y1, x2, y2, color)
    (tw, _), _ = cv2.getTextSize(text, FONT, 1.4, 3)
    cv2.putText(frame, text, (width // 2 - tw // 2, y1 + 55), FONT, 1.4, WHITE, 3, cv2.LINE_AA)

    # Barra de confianza: llena = el modelo está muy seguro; apenas pasado el umbral = ojo.
    p = decision["probs"][prob_key] or 0.0
    bx1, bx2, by = x1 + 20, x2 - 80, y1 + 78
    cv2.rectangle(frame, (bx1, by), (bx2, by + 12), WHITE, 1)
    cv2.rectangle(frame, (bx1, by), (bx1 + int((bx2 - bx1) * p), by + 12), WHITE, -1)
    cv2.putText(frame, f"{p:.2f}", (bx2 + 10, by + 12), FONT, 0.6, WHITE, 2, cv2.LINE_AA)

    others = len(summary.split(" + ")) - 1
    if others:
        cv2.putText(frame, f"+{others}", (x2 - 50, y1 + 30), FONT, 0.8, WHITE, 2, cv2.LINE_AA)
    return summary


def _small_indicator(frame, text, color):
    cx = frame.shape[1] // 2
    cv2.circle(frame, (cx - 60, 40), 10, color, -1)
    cv2.putText(frame, text, (cx - 40, 48), FONT, 0.7, color, 2, cv2.LINE_AA)


def draw_hud(frame, fps, det_ms, decision, age_s):
    """Métricas técnicas, pequeñas y en una esquina: importan a nosotros, no al conductor."""
    lines = [f"{fps:.0f} FPS", f"det {det_ms:.0f} ms"]
    if decision is not None and decision["probs"]:
        lines.append(f"API {decision['latency_ms']:.0f} ms / {age_s:.1f} s")
    x = frame.shape[1] - 230
    _panel(frame, x - 10, 100, frame.shape[1] - 10, 110 + 24 * len(lines), (0, 0, 0), alpha=0.5)
    for k, line in enumerate(lines):
        cv2.putText(frame, line, (x, 122 + 24 * k), FONT, 0.55, WHITE, 1, cv2.LINE_AA)


def draw_debug(frame, memory, light_color, decision):
    """Modo depuración (tecla d): trapecio, calibración, estelas y probabilidades."""
    width = frame.shape[1]
    half = memory.lane_half_width_at(memory.hood_y)
    lane = np.array([(width / 2, memory.horizon_y), (width / 2 + half, memory.hood_y),
                     (width / 2 - half, memory.hood_y)], dtype=np.int32)
    cv2.polylines(frame, [lane], isClosed=True, color=WHITE, thickness=1)
    for y in (memory.horizon_y, memory.hood_y):
        cv2.line(frame, (0, y), (width, y), (255, 0, 255), 1)
    for hist in memory.history.values():
        pts = np.array([obs.bottom_center for obs in hist], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=False, color=YELLOW, thickness=1)
    lines = [f"semaforo: {light_color}"]
    if decision is not None and decision["probs"]:
        lines += [f"{k}: {'-' if p is None else f'{p:.2f}'}" for k, p in decision["probs"].items()]
    for k, line in enumerate(lines):
        cv2.putText(frame, line, (20, 180 + 24 * k), FONT, 0.55, YELLOW, 2, cv2.LINE_AA)


class Beeper:
    """Sonido para avisos urgentes, sin congelar el vídeo y sin pitar sin parar."""

    def __init__(self):
        self._last = 0.0
        self._active = set()  # avisos urgentes activos en el frame anterior

    def update(self, summary):
        active = set(summary.split(" + ")) & SOUNDS.keys() if summary else set()
        started = active - self._active  # los que acaban de EMPEZAR (flanco de subida)
        self._active = active
        now = time.perf_counter()
        if not started or now - self._last < SOUND_COOLDOWN_S:
            return
        self._last = now
        # El más grave de los que empiezan (SOUNDS está ordenado por gravedad).
        sound = next(path for key, path in SOUNDS.items() if key in started)
        # Popen y no run(): lanza afplay (reproductor de macOS) y vuelve AL INSTANTE.
        # run() esperaría a que acabe el sonido y congelaría el vídeo ~0,5 s.
        subprocess.Popen(["afplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
