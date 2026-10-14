"""Llamada a la Decisions API de OpenAI: estado JSON → probabilidad de cada aviso.

Usamos 4 predicados (y no una "choice") para que varios avisos puedan saltar a la vez:
p. ej. un peatón cruzando con el semáforo en rojo. TODO_OK = ningún aviso supera su umbral.
Documentación: https://developers.openai.com/api/docs/guides/decisions (beta: puede cambiar).
"""

import json
import threading
import time
from collections import deque

import openai
from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-6-luna"  # el único disponible en la beta
# En un copiloto, una respuesta que tarda más de esto ya no sirve: mejor no tener decisión.
TIMEOUT_S = 2.0

# Interpretación de cada probabilidad. Entre los dos umbrales no nos atrevemos a decir
# ni "sí" ni "no": es "incierto" (los predicados no traen campo confidence).
# Provisionales: los ajustaremos en la Fase 8 con vídeos etiquetados.
WARN_THRESHOLD = 0.7
UNCERTAIN_THRESHOLD = 0.4
# Nombre del aviso que se muestra, en orden de prioridad (el primero es el más grave).
LABELS = {
    "frenar_ya": "FRENAR_YA",
    "peaton_en_trayectoria": "PEATON_CERCA",
    "semaforo_atencion": "SEMAFORO_AMBAR_ROJO",
    "mantener_distancia": "MANTENER_DISTANCIA",
}
# Cuántas decisiones anteriores le pasamos a la API en historial_decisiones.
HISTORY_LENGTH = 3

# Errores en los que reintentar NO sirve: la clave es mala, no tenemos permiso o la
# petición está mal formada. Seguir llamando solo gastaría y llenaría la pantalla de errores.
FATAL_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError)
# Para el resto (red, timeout, límite de peticiones, fallo del servidor) esperamos y
# reintentamos, cada vez un poco más (backoff), hasta este máximo.
MAX_BACKOFF_S = 8.0

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

# Explica qué significa cada campo del estado: sin esto el modelo tendría que adivinar
# qué es "distancia_rel" o en qué unidades va "ttc_s". Va en inglés como las preguntas,
# aunque los valores del estado sigan en español (el modelo entiende ambos).
CONTEXT = """You are a driving co-pilot. You receive the scene state of a forward-facing dashcam as JSON.
Fields: zona = left/center/right relative to our lane ("centro" = in our path);
distancia_rel = 0 touching our hood, 1 at the horizon (relative, not meters);
acercandose = the object is getting closer;
ttc_s = seconds to collision from how fast the object grows in the image (null if not approaching; noisy and lags behind sudden braking);
ttc_tendencia_s = seconds until distancia_rel reaches 0 at the rate of the last second (more reliable than ttc_s when the vehicle ahead brakes);
lateral = hacia_centro (moving into our path) / alejandose (moving away) / estable;
semaforo = color of the light for our lane (null if none, "desconocido" if unreadable).
historial_decisiones = our previous warnings, most recent last.
Guidance: ttc_s and ttc_tendencia_s are only given for objects in our lane (null otherwise).
Objects outside our lane that are not moving hacia_centro are not collision threats (we are passing them or they are passing us).
If semaforo is null or "desconocido" we could not read any light: then the answer to the traffic light question is no (do not guess the color).
Being very close (low distancia_rel) WITHOUT approaching is not an emergency (e.g. stopped behind a car in traffic): braking now is needed only if something in our path is approaching fast."""

load_dotenv()  # mete OPENAI_API_KEY de .env en el entorno: la clave nunca aparece en el código
# Un único cliente para todo el programa: reutiliza la conexión y ahorra el "calentamiento".
# max_retries=0: el SDK reintenta por su cuenta 2 veces por defecto, lo que convertiría
# nuestro timeout de 2 s en hasta ~6 s + esperas. Los reintentos los hace DecisionWorker,
# que sabe que una decisión tardía no sirve y prefiere pedir una nueva con el estado actual.
_client = OpenAI(max_retries=0)


def decide(state):
    """Devuelve {nombre_del_aviso: probabilidad} para el estado dado.

    Si el modelo rechaza una pregunta ("refusal", un tipo que no está en la guía pero sí en
    el SDK), ese aviso vale None: no sabemos, que no es lo mismo que probabilidad 0.
    Los errores de red y timeouts se dejan pasar hacia arriba (los gestionaremos en el paso 4).
    """
    decision = _client.decisions.create(
        model=MODEL,
        input=CONTEXT + "\n\nScene state:\n" + json.dumps(state, ensure_ascii=False),
        questions=QUESTIONS,
        timeout=TIMEOUT_S,
    )
    return {a.name: (a.probability if a.type == "predicate" else None) for a in decision.answers}


def warm_up():
    """Llamada de prueba al arrancar: la primera tarda 1-3 s (abrir la conexión).
    Mejor pagar ese tiempo antes de empezar el vídeo que en la primera decisión real."""
    decide({"objetos": [], "semaforo": None, "historial_decisiones": []})



def interpret(probs):
    """{aviso: probabilidad} → (estado de cada aviso, resumen para mostrar).

    Cada aviso queda en "si" / "incierto" / "no" (o None si la API lo rechazó).
    Resumen: los avisos activos por prioridad ("FRENAR_YA + SEMAFORO_AMBAR_ROJO"),
    "INCIERTO" si no hay ninguno pero alguno está en la franja dudosa, o "TODO_OK".
    """
    status = {}
    for name, p in probs.items():
        if p is None:
            status[name] = None
        elif p >= WARN_THRESHOLD:
            status[name] = "si"
        elif p >= UNCERTAIN_THRESHOLD:
            status[name] = "incierto"
        else:
            status[name] = "no"
    active = [label for name, label in LABELS.items() if status.get(name) == "si"]
    if active:
        return status, " + ".join(active)
    if "incierto" in status.values() or None in status.values():
        return status, "INCIERTO"
    return status, "TODO_OK"


class DecisionWorker:
    """Pide decisiones a la API en un HILO APARTE para no congelar el vídeo.

    El bucle principal llama a submit(state) cuando quiera (no bloquea) y lee latest()
    en cada frame. Si llegan estados más rápido de lo que responde la API, solo se usa
    el más reciente: una decisión sobre una escena de hace 3 s ya no sirve.
    """

    def __init__(self, min_interval_s=0.5):
        self.min_interval_s = min_interval_s  # no llamar más de ~2 veces por segundo (coste)
        # El candado protege TODO lo que comparten los dos hilos: sin él, el bucle podría
        # leer una decisión a medio escribir (race condition).
        self._lock = threading.Lock()
        self._pending = None  # último estado recibido y aún no enviado
        self._latest = None  # última decisión: probs, status, summary, error, state_t, latency_ms
        self._new_state = threading.Event()  # "despertador" del hilo cuando llega un estado
        # daemon=True: el hilo muere solo al cerrar el programa (si no, lo dejaría colgado).
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, state):
        """Deja un estado para decidir. No espera a la API: vuelve al instante."""
        with self._lock:
            self._pending = state  # si había uno sin enviar, lo sustituye (el viejo ya no sirve)
        self._new_state.set()

    def latest(self):
        """Última decisión disponible (o None si aún no hay ninguna)."""
        with self._lock:
            return self._latest

    def _run(self):
        # Este bucle corre en el otro hilo, para siempre (o hasta un error fatal).
        backoff_s = 0.0
        history = deque(maxlen=HISTORY_LENGTH)  # solo lo usa este hilo: no necesita candado
        while True:
            self._new_state.wait()  # duerme hasta que llegue un estado (sin gastar CPU)
            self._new_state.clear()
            with self._lock:
                state, self._pending = self._pending, None
            if state is None:
                continue
            # El estado ya es "nuestro" (el bucle principal no lo vuelve a tocar),
            # así que podemos añadirle el historial sin candado.
            state["historial_decisiones"] = list(history)

            t0 = time.perf_counter()
            try:
                probs = decide(state)
                status, summary = interpret(probs)
                result = {"probs": probs, "status": status, "summary": summary, "error": None}
                history.append(summary)
                backoff_s = 0.0  # ha ido bien: la próxima vez, sin espera extra
            except FATAL_ERRORS as exc:
                # Reintentar no lo arreglaría: avisamos y paramos el hilo para siempre.
                self._publish(state, t0, {"probs": None, "status": None, "summary": None,
                                          "error": f"FATAL {type(exc).__name__}: revisa la clave/petición"})
                return
            except Exception as exc:  # noqa: BLE001 — red, timeout, rate limit, servidor...
                # Sin esto, un fallo de red mataría el hilo EN SILENCIO. Reintentamos con
                # espera creciente (0,5 → 1 → 2 → 4 → 8 s) para no machacar un servidor caído.
                backoff_s = min(max(2 * backoff_s, 0.5), MAX_BACKOFF_S)
                result = {"probs": None, "status": None, "summary": None,
                          "error": f"{type(exc).__name__} (reintento en {backoff_s:.1f} s)"}
            latency_ms = self._publish(state, t0, result)

            # Respetar el ritmo máximo, más la espera extra si venimos de un error.
            time.sleep(max(0.0, self.min_interval_s - latency_ms / 1000) + backoff_s)

    def _publish(self, state, t0, result):
        latency_ms = 1000 * (time.perf_counter() - t0)
        # Guardamos también el estado evaluado: el vídeo de demo enseña qué se envió.
        result.update(state=state, state_t=state["timestamp_ms"] / 1000, latency_ms=latency_ms)
        with self._lock:
            self._latest = result
        return latency_ms
