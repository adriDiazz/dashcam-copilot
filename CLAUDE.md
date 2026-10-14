# Copiloto con Dashcam — YOLO + OpenAI Decisions

Proyecto personal de visión artificial en tiempo real: un "copiloto" que mira el vídeo de una dashcam, entiende la escena y decide si avisar al conductor.

> ⚠️ Proyecto educativo. No es un sistema de seguridad y no debe usarse para tomar decisiones reales al volante.

---

## 🧑‍🏫 Cómo quiero trabajar contigo (léelo primero, Claude)

Estoy aprendiendo. **No quiero que hagas el proyecto entero.** Quiero construirlo contigo paso a paso y entender cada pieza.

Reglas:

1. **Una fase cada vez.** No avances a la siguiente fase hasta que yo lo diga.
2. **Explica antes de escribir código.** Para cada paso: qué vamos a hacer, por qué, y qué alternativas había.
3. **Código pequeño.** Bloques cortos que yo pueda leer entero. Nada de generar 5 ficheros de golpe.
4. **Déjame intentarlo.** Cuando un paso sea asequible, propónme que lo escriba yo primero y luego revísalo.
5. **Hazme preguntas.** Al terminar cada fase, hazme 2-3 preguntas para comprobar que lo he entendido.
6. **Comenta el código** explicando el _porqué_, no solo el _qué_.
7. **Si algo de la Decisions API de OpenAI no lo sabes seguro, dilo** y miramos la documentación oficial juntos (https://developers.openai.com/api/docs/guides/decisions). Está en beta pública: puede cambiar. No te inventes endpoints ni parámetros.
8. Hablamos en **español**.

---

## 🎯 Objetivo

Dado un vídeo de conducción (o una webcam/móvil en el salpicadero):

1. Detectar en tiempo real coches, motos, peatones, ciclistas, semáforos y señales.
2. Hacer tracking de cada objeto y estimar si se acerca (distancia relativa / tiempo hasta colisión).
3. Convertir la escena en un **estado estructurado (JSON)**.
4. Pedir a la **Decisions API de OpenAI** que decida entre un conjunto cerrado de opciones, con probabilidades.
5. Pintar en pantalla las detecciones + la decisión + la confianza.

---

## 🏗️ Arquitectura

```
Vídeo / cámara
     │  (~30 fps)
     ▼
YOLO (detección) + ByteTrack (tracking)
     │
     ▼
Estimación de riesgo (distancia relativa, velocidad de acercamiento, TTC)
     │
     ▼
Constructor de estado  ──►  JSON de la escena
     │  (~5-10 Hz, no hace falta cada frame)
     ▼
OpenAI Decisions API (decisión tipada + probabilidades)
     │
     ▼
Overlay en pantalla (OpenCV) + log
```

Idea clave: **YOLO son los ojos, Decisions es el cerebro.** Empezamos mandándole solo el estado estructurado (texto), porque es más barato, más rápido y nos obliga a entender la escena. La API también acepta imágenes (base64): en la Fase 8 podemos comparar "solo estado" vs. "estado + frame".

### Decisiones posibles

Usamos **4 preguntas `predicate`** en una sola petición (no una `choice`), para que puedan saltar varios avisos a la vez (p. ej. peatón cruzando + semáforo en rojo):

| Predicado               | Aviso                                    |
| ----------------------- | ---------------------------------------- |
| `frenar_ya`             | Riesgo de colisión inminente             |
| `mantener_distancia`    | Vehículo delante acercándose poco a poco |
| `peaton_en_trayectoria` | Peatón o ciclista en la trayectoria      |
| `semaforo_atencion`     | Semáforo en ámbar o rojo                 |

Cada uno devuelve una `probability` (0-1). `TODO_OK` = ningún aviso supera su umbral. Los predicados no traen `confidence`: "incierto" lo definimos nosotros (p. ej. probabilidad en una franja intermedia). Una respuesta puede ser `refusal` (no está en la guía, sí en el SDK) → ese aviso vale `None`.

### Ejemplo de estado (borrador)

```json
{
  "timestamp_ms": 12345,
  "objetos": [
    {
      "id": 7,
      "clase": "car",
      "zona": "centro",
      "distancia_rel": 0.42,
      "acercandose": true,
      "ttc_s": 2.1
    },
    {
      "id": 12,
      "clase": "person",
      "zona": "derecha",
      "distancia_rel": 0.7,
      "acercandose": false,
      "ttc_s": null
    }
  ],
  "semaforo": "ambar",
  "historial_decisiones": ["TODO_OK", "MANTENER_DISTANCIA"]
}
```

---

## 🧰 Stack

- **Python 3.11+**
- **Ultralytics YOLO** (modelo nano/small para tiempo real)
- **ByteTrack** (viene integrado en Ultralytics)
- **OpenCV** para leer vídeo y pintar el overlay
- **OpenAI Decisions API** (`POST /v1/decisions`, modelo `gpt-6-luna`, SDK Python ≥ 3.26.0) para la decisión
- `python-dotenv` para la API key (nunca en el código ni en git)

Hardware: Mac con Apple Silicon → GPU vía MPS (`device="mps"`). Decidido en la Fase 0.

---

## 📁 Estructura propuesta

```
dashcam-copilot/
├── CLAUDE.md
├── README.md
├── .env.example          # OPENAI_API_KEY=...
├── requirements.txt
├── data/videos/          # vídeos de prueba (no se suben a git)
├── src/
│   ├── capture.py        # leer vídeo / cámara
│   ├── detector.py       # YOLO + tracking
│   ├── risk.py           # distancia relativa, TTC
│   ├── state.py          # escena → JSON
│   ├── decider.py        # llamada a la Decisions API
│   ├── overlay.py        # dibujar en pantalla
│   └── main.py           # bucle principal
└── tests/
```

La crearemos poco a poco, no de golpe.

---

## 🗺️ Fases

### Fase 0 — Preparación

- Entorno virtual, dependencias, comprobar GPU/CPU.
- Conseguir 2-3 vídeos de conducción de prueba (grabados por mí o de datasets abiertos como BDD100K).
- **Entender:** qué es un frame, FPS y por qué importa la latencia.

### Fase 1 — Ver vídeo

- Abrir un vídeo con OpenCV y mostrarlo frame a frame. Medir FPS.
- **Entender:** el bucle de captura, formato BGR, resolución vs. velocidad.

### Fase 2 — Detectar

- YOLO preentrenado (COCO) sobre cada frame. Pintar cajas y etiquetas.
- Filtrar solo las clases que nos interesan.
- **Entender:** qué es una bounding box, confianza, NMS, modelos n/s/m.

### Fase 3 — Seguir (tracking)

- Activar ByteTrack para que cada objeto tenga un ID estable.
- **Entender:** detección vs. tracking, por qué el ID es necesario para saber si algo se acerca.

### Fase 4 — Medir riesgo

- Estimar distancia relativa (por el tamaño/posición de la caja) y velocidad de acercamiento.
- Calcular un TTC aproximado (tiempo hasta colisión). Definir zonas: izquierda/centro/derecha.
- **Entender:** limitaciones de la visión monocular, ruido y suavizado (media móvil).

### Fase 5 — Construir el estado

- Convertir la escena en el JSON de arriba. Decidir qué información es útil y cuál es ruido.
- **Entender:** por qué un buen estado importa más que el modelo.

### Fase 6 — Decidir con la Decisions API

- Leer juntos la documentación y elegir el tipo de pregunta adecuado (`predicate`, `choice` o `score`).
- Llamar a la API a ~1-5 Hz (medir latencia real) (no cada frame), en un hilo aparte para no bloquear el vídeo.
- Gestionar errores, timeouts y confianza baja.
- **Entender:** decisión tipada vs. LLM, latencia, coste, probabilidades calibradas.

### Fase 7 — Mostrar

- Overlay: cajas por color según riesgo, decisión actual, barra de confianza, FPS.
- Opcional: aviso sonoro.

### Fase 8 — Evaluar y mejorar

- Etiquetar a mano unos clips y comparar lo que decide el sistema con lo que debería.
- Comparar la Decisions API con una regla simple (`if ttc < 1.5: FRENAR_YA`). ¿Aporta algo?
- **Entender:** precisión, falsos positivos y por qué un aviso falso también es un problema.

### Extras (si nos animamos)

- Lectura de señales de velocidad.
- Detección de líneas de carril.
- Correr en el móvil o en una Raspberry Pi.
- Grabar una demo para el portfolio.

---

## ✅ Estado actual

- [x] Fase 0
- [x] Fase 1
- [x] Fase 2
- [x] Fase 3
- [x] Fase 4
- [x] Fase 5
- [x] Fase 6
- [x] Fase 7
- [x] Fase 8

(Iré marcando las fases según las terminemos. Claude: consulta esta lista al empezar cada sesión para saber por dónde vamos.)

---

## 📝 Notas y aprendizajes

_Aquí apuntaremos lo que vaya aprendiendo, decisiones que tomemos y problemas que encontremos._

### Fase 0 (2026-10-13)

- **Entorno:** `.venv` con Python 3.13 (de Anaconda). Activar con `source .venv/bin/activate`; comprobar con `which python` que apunta a `.venv`.
- **GPU:** Mac Apple Silicon → PyTorch usa `device="mps"`. Verificado creando un tensor en MPS (no basta con `is_available()`).
- **Vídeos:** usamos el dataset **Nexar Collision Prediction** (Hugging Face, acceso restringido, licencia Nexar Open Data) en vez de BDD100K (ZIPs de ~18 GB). Vídeos 1280×720, ~30 FPS, ~40 s. Se descargan con `scripts/download_videos.py` (requiere `hf auth login`).
- **Etiquetas de Nexar:** `metadata.csv` trae `time_of_event` y `time_of_alert` → útiles en la Fase 8. ⚠️ `light_conditions` **no** distingue día/noche (vídeos de noche marcados como `Normal`): elegimos los de día midiendo el brillo de los frames. Lección: no fiarse de las etiquetas sin mirar muestras.
- **Latencia:** a 30 FPS hay ~33 ms por frame. Si procesar tarda más, mejor saltar frames que acumular retraso. La llamada a la API va en un hilo aparte para no congelar el vídeo.

### Fase 1 (2026-10-13)

- **Bucle de captura** en `src/capture.py`: `VideoCapture` → `read()` → dibujar → `imshow` → `waitKey`. Al final, `release()` + `destroyAllWindows()` (en macOS, si no, la ventana se cuelga).
- `VideoCapture` no da error si la ruta está mal → comprobar `isOpened()`. `cv2.CAP_PROP_*` son identificadores; el valor se pide con `cap.get()`.
- `frame.shape` = `(alto, ancho, 3)` (filas × columnas × canales BGR), al revés que "1280×720".
- **Todo se pinta antes de `imshow`.** `waitKey` es obligatorio para que la ventana se refresque.
- **FPS:** medir con `time.perf_counter()` restando la vuelta anterior (dos variables: `prev_time` y `elapsed`). Suavizar con `deque(maxlen=30)` promediando **tiempos**, no FPS.
- **Medición:** ~60 FPS con ventana vs. ~2.000 sin ella → el cuello de botella es la pantalla (60 Hz), no leer el vídeo. Para medir YOLO, cronometrarlo por separado.

### Fase 2 (2026-10-14)

- **YOLO en el bucle** (`src/capture.py`): modelo cargado una vez fuera del bucle; `model(frame, device="mps", classes=CLASSES, conf=0.25, verbose=False)`; `results[0].plot()` para dibujar (en la Fase 7 dibujaremos nosotros).
- **Clases COCO usadas:** 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck, 9 traffic light, 11 stop sign. COCO no da el color del semáforo ni otras señales.
- **Caja** `[x1, y1, x2, y2]` en píxeles (esquina sup. izq. → inf. der.). Caja más grande y más abajo ⇒ objeto más cerca.
- **Umbral de confianza:** 0.35 quita falsos positivos (árbol → persona) pero pierde el 71 % de los semáforos (son pequeños → baja confianza). Un único umbral no sirve para todas las clases → umbral por clase en la Fase 4.
- **Comparación** (`scripts/compare_models.py`, vídeo 00021, MPS): 26n 5,4 ms · **26s 16,7 ms** · 11n 5,0 ms · 26n en CPU 19,9 ms. 26s detecta ~3× más semáforos y +35 % personas y con más confianza → **usamos `yolo26s.pt`**. Ojo: más detecciones ≠ mejor sin etiquetas (Fase 8).
- **Problemas observados:** cajas que parpadean (YOLO no tiene memoria entre frames → Fase 3), árboles detectados como persona, y **al chocar el coche deja de detectarse** (demasiado cerca/cortado). ⚠️ Perder un objeto que venía acercándose rápido NO significa "todo OK": el aviso debe saltar antes (TTC) y el sistema necesita memoria.
- El primer frame tarda ~1 s (calentamiento de la GPU): no contarlo al medir.

### Fase 3 (2026-10-14)

- **Tracking** con `model.track(frame, persist=True, tracker="bytetrack.yaml", ...)`. `persist=True` conserva la memoria entre llamadas. Requiere `lap` (añadido a requirements).
- **Sin `conf=` propio:** `model.track()` usa `conf=0.1` porque ByteTrack aprovecha las detecciones dudosas (0.1–0.25) para continuar objetos ya seguidos. Umbrales reales en `bytetrack.yaml` (`track_high_thresh` 0.25, `track_low_thresh` 0.1, `new_track_thresh` 0.25, `track_buffer` 30).
- `r.boxes.id` puede ser `None`. Los IDs no son consecutivos (se reservan para candidatos no confirmados): solo importa que no cambien.
- **Estelas:** `trails = {id: deque(maxlen=30)}` con el punto inferior central de la caja; limpieza con `last_seen` para no crecer sin fin.
- **Resultado (vídeo 00021, yolo26s):** el coche de delante mantiene el **ID 591 durante todo el choque**. Su caja pasa de ~300 px (alerta, frame ~563) a ~500 px (choque, frame ~601) y el crecimiento se acelera → base del TTC (`ancho / crecimiento por segundo`: ~5 s → ~1,2 s).
- 58 de 129 tracks duran < 0,5 s (falsos positivos, objetos lejanos o cambios de ID) → en la Fase 4, ignorar objetos con poca antigüedad.
- BoT-SORT (con compensación del movimiento de cámara) queda como alternativa si hay cambios de ID en curvas.
- ⚠️ El shell de Claude no tiene el `.venv` activo: ejecutar con `.venv/bin/python`. Anaconda tiene otra versión de Ultralytics (8.4.56) que pelea por el fichero de ajustes.

### Fase 4 (2026-10-14)

- **`src/risk.py`** (`SceneMemory`): no sabe nada de YOLO/OpenCV; recibe `(id, clase, caja)` + tiempo. Historial por ID (`deque` de `Observation`), olvida tras 1 s sin verse, `is_mature` = ≥ 10 observaciones.
- **Tiempo del vídeo** (`frame / fps`), no reloj real: reproducimos a ~60 FPS y el TTC saldría la mitad.
- **Calibración por vídeo** (`HORIZON_Y`, `HOOD_Y`, en `capture.py` y `scripts/eval_ttc.py`): cambia mucho entre cámaras (00021: 300/530, 01074: 400/560, 00015: 450/580).
- **Zonas:** trapecio con forma de carril (medio ancho 25 % en el capó, converge en el centro del horizonte). Las franjas verticales metían en "centro" coches lejanos del carril contrario. Solo para clases que van por el suelo (no semáforos).
- **Distancia relativa:** `(HOOD_Y − y_inferior) / (HOOD_Y − HORIZON_Y)`, recortada a [0, 1]. Sirve para comparar, no son metros.
- **TTC = 1 / crecimiento relativo del tamaño**, con tamaño = √(ancho × alto) y crecimiento = pendiente de una recta sobre log(tamaño) en los últimos 10 frames (suavizado). No necesita metros. Solo si crece > 5 %/s, TTC ≤ 10 s y d ≤ 0,8 (lo lejano es ruido).
- **Fallos corregidos:** YOLO detecta nuestro propio capó como `car` → se descartan cajas con centro bajo `HOOD_Y`; oclusiones parciales de coches lejanos daban TTC falsos.
- **Evaluación** (`scripts/eval_ttc.py`, avisos TTC < 2 s en centro): 01050 → 0, 01074 → 4, 01046 → 12 (cámara apuntando al cielo), 00021 → choque detectado tarde (19,6 s vs. alerta Nexar 18,8 s), 00015 → **a tiempo** (17,8 s vs. 17,7 s).
- **Limitaciones:** el TTC supone velocidad constante (llega tarde con frenazos bruscos); calibración fija no aguanta cuestas ni cámaras mal montadas (solución: estimar el horizonte con las líneas del carril); TTC bajo ≠ peligro (un coche de frente por el otro carril se acerca rápido pero su trayectoria no cruza la nuestra).

### Fase 5 (2026-10-14)

- **`src/state.py`** (`build_state(memory, t, traffic_light, decision_history)`): solo objetos vistos en este instante, maduros y que van por el suelo; **relevantes** si están en el centro, a d < 0,4 o son peatón/ciclista; ordenados por peligro (TTC, luego distancia); máximo 5. Clases simplificadas (`vehiculo`/`peaton`/`ciclista`), valores redondeados. Sin cajas en píxeles ni confianza de YOLO. ~7 detecciones → ~1-3 objetos, ~240-400 caracteres.
- **Movimiento lateral** (`risk.py`: `lane_offset`, `lateral_motion`): offset en "medios carriles" (dividir por el medio ancho del trapecio a esa altura corrige la perspectiva; ±2 ≈ carril de al lado). Pendiente de |offset| en 10 frames → `hacia_centro`/`alejandose`/`estable` (umbral 0,3/s). Descarta el carril contrario y detecta al peatón que cruza. Supone que vamos recto.
- **Semáforo** (`src/traffic_light.py`): HSV (tono = qué color; saturación y brillo = luz encendida), píxeles cerca del máximo brillo del recorte, el semáforo más grande y votación de 10 frames. Funciona con luces claras (sobre todo de noche); falla con carcasas amarillas al sol, centros quemados en blanco y en cruces (elige el semáforo equivocado). Usar `r.orig_img`, no el frame dibujado.
- **Estados reales** (`scripts/print_states.py`): en 00021 la `distancia_rel` baja regular (0,45 → 0,09) mientras el TTC salta (5,1 → 9,7 → null → 8,6 → 1,7). En 00000 (furgoneta lateral) la señal es débil y ruidosa: un vehículo cortado por el borde de la imagen "crece" al entrar, aunque no se acerque.
- 🔑 **Un estado es una foto; el peligro está en la tendencia.** Con solo el estado de la alerta (TTC 8,6) se decidiría `MANTENER_DISTANCIA`. En la Fase 6 hay que mandar algo del pasado (campo de cambio de distancia o los últimos estados).

### Fase 6 (2026-10-14)

- **Clave:** `.env` (permisos 600, en `.gitignore`), `.env.example` sin clave. `load_dotenv()` + `OpenAI()` la leen del entorno; nunca aparece en el código ni en el chat.
- **SDK:** `client.decisions.create(model="gpt-6-luna", input=..., questions=..., timeout=...)` (la guía solo trae `curl`; lo comprobamos inspeccionando el SDK 3.29). Las respuestas pueden ser de tipo **`refusal`** (no está en la guía) → ese aviso vale `None`.
- **4 predicados** en una petición (`frenar_ya`, `mantener_distancia`, `peaton_en_trayectoria`, `semaforo_atencion`) + un contexto que explica cada campo del estado.
- 🔑 **Feature engineering:** con `distancia_hace_1s` el modelo NO hizo la cuenta (`frenar_ya` 0,12 → 0,15); con **`ttc_tendencia_s`** ya calculado, 0,93. Dar las conclusiones calculadas, no los datos en bruto. `ttc_tendencia_s` (recta sobre ~1 s de distancias) es mucho más estable que el TTC por tamaño.
- **Latencia** ~250-550 ms (la primera llamada 1-3 s → `warm_up()` al arrancar). **Coste** ~870 tokens/llamada ≈ 0,63 $/hora a 2 llamadas/s.
- **`DecisionWorker`** (hilo aparte, `daemon=True`): `submit(state)` no bloquea, `latest()` devuelve la última decisión. `Lock` para lo compartido, `Event` para despertar al hilo, "el estado más reciente gana", máximo 2 llamadas/s. El estado se construye en el hilo principal: **nunca compartir `SceneMemory`** entre hilos (race condition).
- **Robustez:** vídeo a velocidad real (`waitKey` espera lo que falta; sin esto la API parecía el doble de lenta); decisiones con edad > 1,5 s no se muestran; umbrales provisionales `si` ≥ 0,7, `incierto` 0,4-0,7; errores fatales (clave, permisos, petición) paran el hilo, transitorios (red, timeout, rate limit, servidor) reintentan con backoff 0,5 → 8 s; `max_retries=0` en el cliente (el SDK reintenta 2 veces por defecto); historial de los 3 últimos resúmenes.
- **Resultado (00021, velocidad real):** `FRENAR_YA` en pantalla a los 18,0 s, antes de la alerta de Nexar (18,8 s) y 2 s antes del choque. Pendiente: ¿falsas alarmas en vídeos sin choque? → Fase 8.

### Fase 7 (2026-10-14)

- **`src/overlay.py`**: cajas por riesgo calculado por nosotros en cada frame (gris = fuera del estado, verde, ámbar TTC ≤ 5 s, rojo TTC < 2,5 s), banner con el aviso más grave (FRENA YA parpadea) + barra de confianza + "+N", indicador pequeño para OK/incierto/sin decisión, HUD en una esquina, modo depuración con la tecla `d`. Paneles semitransparentes con `cv2.addWeighted`. Ya no se usa `r.plot()`.
- **Sonido** (`Beeper`): `afplay` con `subprocess.Popen` (no bloquea; `run()` congelaría el vídeo). Solo FRENAR_YA y PEATON_CERCA, y **solo cuando el aviso empieza** (flanco de subida): con "uno cada 2 s" sonaba 11 veces seguidas tras el choque.
- `capture.py` acepta el vídeo por argumento (`python src/capture.py positive/00015`) con un diccionario `CALIBRATION` por vídeo.
- ⚠️ **El overlay destapó el gran problema:** FRENA YA salta al **adelantar o pasar cerca de coches por los lados** (su caja crece rápido y se corta en el borde → TTC bajo). 01050 (sin incidente): 7 pitidos; 00015: aviso bueno a los 19,2 s (choque 19,7 s) pero 3 falsos más. → Fase 8.

### Fase 8 (2026-10-14)

- **`scripts/eval_alerts.py`**: caché de detecciones (todas las versiones ven las mismas cajas), API cada 0,5 s de vídeo con reintentos, episodios de FRENAR_YA, acierto = aviso entre 4 s antes del choque y el choque; en negativos todo aviso es falsa alarma; anota el "culpable" de cada una.
- **Arreglos, medidos uno a uno** (choques avisados 3/3 en todos): línea base 8,9 falsas/min → **B** cajas cortadas por el borde sin TTC/lateral 6,4 → **A** TTC solo en nuestro carril 5,0 (la furgoneta de 00000 pasa de 3,3 a 0,4 s de antelación) → **D** TTC por tamaño solo hasta d=0,6: 3,6 → **C** contexto ("cerca sin acercarse no es emergencia", "fuera del carril y sin meterse no es amenaza"): **2,8/min**. Cada arreglo cuesta antelación. Con 6 vídeos, diferencias pequeñas pueden ser ruido (la API no es determinista).
- **Semáforo inventado:** con `semaforo: null` respondía rojo con 1,00 → frase en el contexto ("si es null o desconocido, la respuesta es no"). El overlay ahora aplica el mismo criterio de TTC que el estado.
- **Demo para LinkedIn** (`scripts/render_demo.py`, 1080×1080, grabada en directo; `render_when_ready.py` comprueba la salud de la API y repite si hay huecos). La beta a veces iba lenta (timeouts de 2 s): el copiloto se queda sin decisión → un sistema real necesitaría una regla local de respaldo.
- **Decisions API por dentro:** 0 tokens de salida (no genera texto). Arquitectura no publicada (ni Jev de TypeSafe AI ni OpenAI).
- No llegamos a comparar con una regla simple (`ttc < 1.5`): el objetivo era probar la API.
