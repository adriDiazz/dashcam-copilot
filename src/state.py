"""Escena → estado JSON: lo ÚNICO que verá la Decisions API.

Filtra, ordena y resume lo que sabe SceneMemory (risk.py). Un buen estado es corto
y centrado en lo peligroso: más objetos = más coste, más latencia y más ruido.
"""

import math

# Máximo de objetos en el estado. Con más, lo importante se diluye entre lo irrelevante.
MAX_OBJECTS = 5
# Un objeto fuera del carril solo es relevante si está más cerca que esto.
NEAR_DISTANCE = 0.4

# Simplificamos las clases de COCO: para decidir da igual "car" o "truck"
# (YOLO a veces duda entre ellas, como con la pickup roja).
CLASS_GROUPS = {
    "car": "vehiculo",
    "truck": "vehiculo",
    "bus": "vehiculo",
    "motorcycle": "vehiculo",
    "person": "peaton",
    "bicycle": "ciclista",
}
# Peatones y ciclistas siempre son relevantes: son vulnerables y pueden cruzar.
VULNERABLE = {"peaton", "ciclista"}


def _describe(memory, track_id):
    """Un objeto como diccionario, con valores redondeados (más decimales = solo ruido)."""
    ttc = memory.ttc(track_id)
    ttc_trend = memory.ttc_trend(track_id)
    # El TTC solo significa "choque" en NUESTRO carril. Fuera del trapecio, un coche que
    # adelantamos "crece" al acercarnos de lado y daba TTC < 1 s (Fase 8: la mayoría de
    # falsos FRENAR_YA). Si se está metiendo, lo dice "lateral" y al entrar tendrá TTC.
    if memory.zone(track_id) != "centro":
        ttc = ttc_trend = None
    return {
        "id": track_id,
        "clase": CLASS_GROUPS[memory.history[track_id][-1].cls],
        "zona": memory.zone(track_id),
        "distancia_rel": round(memory.distance_rel(track_id), 2),
        "acercandose": memory.is_approaching(track_id),
        "ttc_s": round(ttc, 1) if ttc is not None else None,
        # La tendencia YA calculada: en la prueba, con este dato frenar_ya pasó de 0,15 a 0,93.
        "ttc_tendencia_s": round(ttc_trend, 1) if ttc_trend is not None else None,
        # Con el TTC sabemos CUÁNDO llegaría; con esto, si viene HACIA nuestra trayectoria.
        "lateral": memory.lateral_motion(track_id),
    }


def _is_relevant(obj):
    return obj["zona"] == "centro" or obj["distancia_rel"] < NEAR_DISTANCE or obj["clase"] in VULNERABLE


def _danger_key(obj):
    # Ordenar por peligro: primero el menor de los dos TTC, luego por cercanía.
    # math.inf manda al final a los que no se acercan.
    ttcs = [v for v in (obj["ttc_s"], obj["ttc_tendencia_s"]) if v is not None]
    return (min(ttcs, default=math.inf), obj["distancia_rel"])


def build_state(memory, t, traffic_light=None, decision_history=()):
    """Estado de la escena en el instante t (segundos de vídeo).

    traffic_light: color del semáforo que nos afecta (TrafficLightMonitor) o None si no hay.
    """
    objects = []
    for track_id, hist in memory.history.items():
        # Solo lo visto EN ESTE instante, maduro (los recién nacidos suelen ser ruido)
        # y de las clases que van por el suelo (los semáforos van aparte).
        if hist[-1].t != t or not memory.is_mature(track_id) or hist[-1].cls not in CLASS_GROUPS:
            continue
        obj = _describe(memory, track_id)
        if _is_relevant(obj):
            objects.append(obj)

    objects.sort(key=_danger_key)
    return {
        "timestamp_ms": round(t * 1000),
        "objetos": objects[:MAX_OBJECTS],
        "semaforo": traffic_light,  # "rojo" / "ambar" / "verde" / "desconocido" / None
        "historial_decisiones": list(decision_history),  # lo rellenará la Fase 6
    }
