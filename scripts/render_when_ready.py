"""Genera el vídeo de demo EN DIRECTO cuando la API va bien, y lo repite si sale con huecos.

La Decisions API (beta) a veces va lenta: entonces el copiloto se queda sin decisión y el
vídeo sale con huecos. Este script espera a que la API esté sana, graba y valida.

Uso (desde la raíz del proyecto): python scripts/render_when_ready.py
"""

import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, "src")
import decider  # noqa: E402

MAX_ATTEMPTS = 6
WAIT_BETWEEN_S = 600  # 10 minutos entre intentos
HEALTH_CALLS = 15
MIN_OK = 14  # como mucho 1 fallo de 15
MAX_MEDIAN_MS = 500
MAX_GAP = 0.05  # cada escena, como mucho un 5 % de fotogramas sin decisión
BEST_PATH = "outputs/demo_linkedin_mejor_intento.mp4"


def api_health():
    state = {"timestamp_ms": 0, "objetos": [], "semaforo": "rojo", "historial_decisiones": []}
    latencies = []
    for _ in range(HEALTH_CALLS):
        t0 = time.perf_counter()
        try:
            decider.decide(state)
            latencies.append(1000 * (time.perf_counter() - t0))
        except Exception:  # noqa: BLE001 — aquí solo contamos fallos
            pass
    median = statistics.median(latencies) if latencies else float("inf")
    return len(latencies), median


best = None  # (peor hueco, intento): por si ninguno pasa, guardamos el menos malo
for attempt in range(1, MAX_ATTEMPTS + 1):
    ok, median = api_health()
    print(f"[intento {attempt}] salud API: {ok}/{HEALTH_CALLS} ok, mediana {median:.0f} ms", flush=True)
    if ok >= MIN_OK and median <= MAX_MEDIAN_MS:
        out = subprocess.run([sys.executable, "scripts/render_demo.py"], capture_output=True, text=True)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("HUECOS:")), None)
        if line is None:
            print(f"[intento {attempt}] el render falló:\n{out.stderr[-2000:]}", flush=True)
        else:
            gaps = [float(x) for x in line.split()[1:]]
            print(f"[intento {attempt}] huecos por escena: {', '.join(f'{100 * g:.1f}%' for g in gaps)}", flush=True)
            if best is None or max(gaps) < best[0]:
                best = (max(gaps), attempt)
                shutil.copy("outputs/demo_linkedin.mp4", BEST_PATH)
            if max(gaps) <= MAX_GAP:
                print(f"ACEPTADO en el intento {attempt}: outputs/demo_linkedin.mp4", flush=True)
                sys.exit(0)
    if attempt < MAX_ATTEMPTS:
        print(f"[intento {attempt}] no vale; espero {WAIT_BETWEEN_S // 60} min", flush=True)
        time.sleep(WAIT_BETWEEN_S)

print(f"NINGUNO PASÓ. Mejor intento: {best[1]} (peor escena {100 * best[0]:.1f}% de huecos) → {BEST_PATH}"
      if best else "NINGUNO PASÓ y la API no estuvo sana en ningún intento.", flush=True)
sys.exit(1)
