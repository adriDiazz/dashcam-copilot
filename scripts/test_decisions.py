"""Primera prueba de la Decisions API: estados escritos a mano, 4 predicados, latencia.

Uso (desde la raíz del proyecto): python scripts/test_decisions.py
Necesita OPENAI_API_KEY en .env. Cada llamada cuesta una fracción de céntimo.
"""

import json
import statistics
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # lee .env y mete OPENAI_API_KEY en el entorno; el código nunca ve la clave escrita
client = OpenAI()  # coge OPENAI_API_KEY del entorno automáticamente

MODEL = "gpt-6-luna"  # el único disponible en la beta (según la documentación)
CALLS_PER_SCENE = 3  # para medir la latencia con algo más que una muestra

# Un predicado por aviso: cada uno tiene su propia probabilidad y pueden saltar varios a la vez.
QUESTIONS = [
    {
        "type": "predicate",
        "name": "frenar_ya",
        "instructions": "Is there an imminent collision risk that requires braking right now?",
    },
    {
        "type": "predicate",
        "name": "mantener_distancia",
        "instructions": "Is a vehicle ahead in our lane getting closer, so the driver should keep more distance?",
    },
    {
        "type": "predicate",
        "name": "peaton_en_trayectoria",
        "instructions": "Is a pedestrian or cyclist in or moving into our path?",
    },
    {
        "type": "predicate",
        "name": "semaforo_atencion",
        "instructions": "Is the traffic light that applies to us amber or red?",
    },
]

# Contexto fijo: explica qué significa cada campo. Sin esto, el modelo tendría que adivinar
# qué es "distancia_rel" o en qué unidades va "ttc_s".
CONTEXT = """You are a driving co-pilot. You receive the scene state of a forward-facing dashcam as JSON.
Fields: zona = left/center/right relative to our lane ("centro" = in our path);
distancia_rel = 0 touching our hood, 1 at the horizon (relative, not meters);
acercandose = the object is getting closer; ttc_s = estimated seconds to collision at constant speed (null if not approaching; it is noisy and lags behind sudden braking);
lateral = hacia_centro (moving into our path) / alejandose (moving away) / estable;
distancia_hace_1s = distancia_rel one second ago (the trend); semaforo = color of the light for our lane (null if none)."""

# Momento de la alerta de Nexar en 00021 (19 s): el coche de delante está frenando.
STATE_ALERT = {
    "objetos": [{"id": 591, "clase": "vehiculo", "zona": "centro", "distancia_rel": 0.19,
                 "acercandose": True, "ttc_s": 8.6, "lateral": "estable"}],
    "semaforo": None,
}
# El mismo estado, más la tendencia: hace 1 s la distancia era 0.30 (bajó 0.11 en 1 s).
STATE_ALERT_TREND = json.loads(json.dumps(STATE_ALERT))
STATE_ALERT_TREND["objetos"][0]["distancia_hace_1s"] = 0.30
# Una escena tranquila, para ver que no avisa por todo.
STATE_CALM = {
    "objetos": [{"id": 12, "clase": "vehiculo", "zona": "derecha", "distancia_rel": 0.7,
                 "acercandose": False, "ttc_s": None, "lateral": "estable", "distancia_hace_1s": 0.7}],
    "semaforo": "verde",
}

SCENES = {
    "alerta 00021 SIN tendencia": STATE_ALERT,
    "alerta 00021 CON tendencia": STATE_ALERT_TREND,
    "escena tranquila": STATE_CALM,
}

for label, state in SCENES.items():
    latencies = []
    for _ in range(CALLS_PER_SCENE):
        t0 = time.perf_counter()
        decision = client.decisions.create(
            model=MODEL,
            input=CONTEXT + "\n\nScene state:\n" + json.dumps(state, ensure_ascii=False),
            questions=QUESTIONS,
            timeout=5.0,  # en un copiloto, una respuesta que tarda más de 5 s ya no sirve
        )
        latencies.append(1000 * (time.perf_counter() - t0))

    print(f"\n=== {label}")
    for a in decision.answers:
        # Además de predicate/choice/score, la API puede devolver "refusal" (no está en la guía,
        # lo vimos en el SDK): no tiene probability, así que hay que comprobar el tipo.
        value = f"{a.probability:.2f}" if a.type == "predicate" else a.type
        print(f"    {a.name:<22} {value}")
    print(f"    latencia: {', '.join(f'{x:.0f}' for x in latencies)} ms (mediana {statistics.median(latencies):.0f})"
          f" | tokens de entrada: {decision.usage.input_tokens}")
