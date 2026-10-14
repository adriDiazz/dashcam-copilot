"""Riesgo de cada objeto seguido: historial, zona, (luego) distancia relativa y TTC.

No sabe nada de YOLO ni de OpenCV: recibe cajas con ID y tiempo, y devuelve números.
Así se puede probar y cambiar sin tocar el bucle de vídeo.
"""

import math
import statistics
from collections import deque
from dataclasses import dataclass

# Cuántas observaciones guardamos por objeto: 30 frames ≈ 1 s de vídeo a 30 FPS.
HISTORY_LENGTH = 30
# Por debajo de esta edad (en frames) no calculamos riesgo: casi la mitad de los
# objetos viven < 0,5 s (falsos positivos, parpadeos) y sin historia no hay TTC.
MIN_AGE = 10
# Si un objeto lleva más de esto sin verse, lo olvidamos (se fue o el tracker lo perdió).
FORGET_AFTER_S = 1.0

# Zona "centro" = un trapecio con forma de carril: ancho a la altura del capó y que se
# estrecha hasta un punto en el horizonte (centro de la imagen). Con franjas verticales,
# los coches lejanos del carril contrario caían en "centro" por la perspectiva y daban
# falsas alarmas de TTC. Medio ancho del trapecio en el capó, en fracción del ancho de imagen:
LANE_HALF_WIDTH = 0.25

# Solo lo que va por el SUELO tiene zona y distancia. Un semáforo cuelga sobre la calzada:
# su posición en la imagen no dice si está "en mi carril" ni a qué distancia.
GROUND_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

# TTC: cuántas observaciones usamos para medir cómo crece la caja. 10 frames ≈ 0,33 s.
# Más = menos ruido, pero reacciona más tarde a un frenazo (y llega tarde al aviso).
TTC_WINDOW = 10
# Por debajo de este crecimiento (fracción por segundo) consideramos que NO se acerca:
# el ancho tiembla unos píxeles aunque el coche esté quieto. 0.05 = crecer un 5 % por segundo.
MIN_GROWTH_RATE = 0.05
# Un TTC mayor que esto no aporta nada (y con crecimientos minúsculos sale enorme).
MAX_TTC_S = 10.0
# Los objetos lejanos ocupan pocos píxeles: un temblor de 3 px o que otro coche los tape
# a medias cambia su tamaño un 20-50 % y da TTC falsos de < 1 s. Por encima de esta
# distancia relativa no calculamos TTC (además, algo tan lejos no es una emergencia).
MAX_DIST_FOR_TTC = 0.6  # era 0.8; Fase 8: coches a d≈0,7 daban TTC falsos de 1-2 s
# Movimiento lateral: velocidad mínima (en "medios carriles" por segundo) para decir que
# un objeto va hacia el centro o se aleja. Por debajo, el temblor de la caja: "estable".
MIN_LATERAL_SPEED = 0.3
# TTC por tendencia de la distancia: usamos ~1 s de historia (todo el deque) y exigimos
# al menos medio segundo de datos. Por debajo de esta caída (distancia_rel por segundo)
# consideramos que la distancia no está bajando de verdad.
MIN_TREND_SPAN_S = 0.5
MIN_DISTANCE_DROP_RATE = 0.02
# Una caja a menos de esto (px) del borde izquierdo/derecho está CORTADA: el objeto se sale
# de la imagen y YOLO solo ve un trozo. Su tamaño, posición y centro son falsos (Fase 8:
# era la causa de la mayoría de falsos FRENAR_YA al adelantar). El borde de abajo no cuenta:
# el coche del choque de 00021 llega pegado al capó y perderíamos justo el aviso bueno.
EDGE_MARGIN_PX = 5


@dataclass
class Observation:
    """Lo que sabemos de un objeto en UN frame."""

    t: float  # segundos de VÍDEO (frame / fps), no reloj real: el TTC debe usar el tiempo grabado
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def size(self):
        # Media geométrica de ancho y alto: si una oclusión o un giro cambian solo
        # una de las dos medidas, el tamaño varía la mitad que usando solo el ancho.
        return math.sqrt(max(self.width, 1.0) * max(self.y2 - self.y1, 1.0))

    @property
    def bottom_center(self):
        # Donde el objeto toca la carretera: es lo que indica su posición real.
        return ((self.x1 + self.x2) / 2, self.y2)


class SceneMemory:
    """Recuerda la historia reciente de cada objeto por su ID de tracking."""

    def __init__(self, frame_width, horizon_y, hood_y):
        self.frame_width = frame_width
        # Calibración de la cámara (en píxeles): dependen de la altura y el ángulo a los
        # que esté montada, así que cambian en cada vídeo. Ver HORIZON_Y/HOOD_Y en capture.py.
        self.horizon_y = horizon_y  # donde la carretera se pierde (distancia "infinita")
        self.hood_y = hood_y  # donde empieza el capó (lo más cerca que se puede ver)
        self.history = {}  # {track_id: deque[Observation]}

    def update(self, t, detections):
        """detections: lista de (track_id, clase, (x1, y1, x2, y2)) del frame actual."""
        for track_id, cls, (x1, y1, x2, y2) in detections:
            # Si el centro de la caja está por debajo del inicio del capó, es NUESTRO coche:
            # YOLO detecta a veces el propio capó como "car", y su caja temblando da TTC falsos.
            if (y1 + y2) / 2 > self.hood_y:
                continue
            obs = Observation(t, cls, x1, y1, x2, y2)
            self.history.setdefault(track_id, deque(maxlen=HISTORY_LENGTH)).append(obs)

        # Olvidar los que no vemos hace rato; si no, el diccionario crece sin fin.
        for track_id in [i for i, h in self.history.items() if t - h[-1].t > FORGET_AFTER_S]:
            del self.history[track_id]

    def is_mature(self, track_id):
        # Usamos len(history) como "edad": cuenta cuántas veces lo hemos visto.
        return len(self.history[track_id]) >= MIN_AGE

    def zone(self, track_id):
        """'izquierda' / 'centro' / 'derecha', o None si el objeto no va por el suelo."""
        obs = self.history[track_id][-1]
        if obs.cls not in GROUND_CLASSES:
            return None
        x, y = obs.bottom_center
        offset = x - self.frame_width / 2  # negativo = a la izquierda del centro
        if abs(offset) <= self.lane_half_width_at(y):
            return "centro"
        return "izquierda" if offset < 0 else "derecha"

    def lane_half_width_at(self, y):
        """Medio ancho (px) del trapecio "centro" a la altura y: 0 en el horizonte, máximo en el capó.

        Suponemos que la cámara mira recto, así que el carril converge en el centro de la imagen.
        """
        frac = (y - self.horizon_y) / (self.hood_y - self.horizon_y)
        return max(frac, 0.0) * LANE_HALF_WIDTH * self.frame_width

    def distance_rel(self, track_id):
        """0 = pegado al capó, 1 = en el horizonte. None si no va por el suelo.

        Usa dónde toca el objeto la carretera (borde inferior de la caja): en una dashcam,
        cuanto más cerca está algo, más abajo aparece. No son metros: solo sirve para
        comparar objetos entre sí, y depende de que la carretera sea más o menos plana.
        """
        obs = self.history[track_id][-1]
        if obs.cls not in GROUND_CLASSES:
            return None
        return self._distance_of(obs)

    def _distance_of(self, obs):
        _, y = obs.bottom_center
        d = (self.hood_y - y) / (self.hood_y - self.horizon_y)
        # Recortamos a [0, 1]: una caja puede pasar del capó (coche pegado y cortado)
        # o quedar por encima del horizonte (cuesta arriba, calibración imperfecta).
        return min(max(d, 0.0), 1.0)

    def growth_rate(self, track_id):
        """Cuánto crece la caja por segundo, en fracción (0.2 = +20 %/s). None si no se puede medir.

        Ajustamos una recta a log(tamaño) frente al tiempo con las últimas TTC_WINDOW
        observaciones. La pendiente es el crecimiento relativo, y la recta "promedia"
        el temblor de las cajas: es nuestro suavizado.
        """
        hist = self.history[track_id]
        if len(hist) < TTC_WINDOW or hist[-1].cls not in GROUND_CLASSES:
            return None
        if self.distance_rel(track_id) > MAX_DIST_FOR_TTC:
            return None
        recent = list(hist)[-TTC_WINDOW:]
        if any(self._truncated(obs) for obs in recent):
            return None
        times = [obs.t for obs in recent]
        # log porque nos interesa el crecimiento RELATIVO: pasar de 100 a 110 px
        # significa lo mismo (+10 %) que pasar de 300 a 330 px.
        log_sizes = [math.log(obs.size) for obs in recent]
        return statistics.linear_regression(times, log_sizes).slope

    def is_approaching(self, track_id):
        rate = self.growth_rate(track_id)
        return rate is not None and rate > MIN_GROWTH_RATE

    def ttc(self, track_id):
        """Segundos hasta el choque si nada cambia, o None si no se acerca.

        El tamaño en la imagen es inversamente proporcional a la distancia, así que
        tamaño / (crecimiento del tamaño por segundo) = distancia / velocidad de acercamiento.
        Con el crecimiento relativo r (ver growth_rate) eso es simplemente 1 / r.
        No hacen falta metros ni el tamaño real del coche.
        """
        if not self.is_approaching(track_id):
            return None
        ttc = 1.0 / self.growth_rate(track_id)
        return ttc if ttc <= MAX_TTC_S else None

    def lane_offset(self, obs):
        """Posición lateral en "medios carriles": 0 = en el centro de nuestro carril,
        ±1 = en el borde del trapecio, ±2 ≈ en el carril de al lado.

        Dividir por el medio ancho del trapecio a esa altura corrige la perspectiva:
        un coche que va recto por el carril de al lado mantiene su valor aunque se acerque
        (en píxeles, en cambio, se "abriría" hacia el borde de la imagen al acercarse).
        """
        x, y = obs.bottom_center
        half = self.lane_half_width_at(y)
        if half < 1:  # en el horizonte el trapecio mide 0: no se puede dividir
            return None
        return (x - self.frame_width / 2) / half

    def lateral_motion(self, track_id):
        """'hacia_centro' / 'alejandose' / 'estable', o None si no se puede medir."""
        hist = self.history[track_id]
        if len(hist) < TTC_WINDOW or hist[-1].cls not in GROUND_CLASSES:
            return None
        # Igual que con el TTC: lo lejano tiene pocos píxeles y la medida es puro ruido.
        if self.distance_rel(track_id) > MAX_DIST_FOR_TTC:
            return None
        recent = list(hist)[-TTC_WINDOW:]
        if any(self._truncated(obs) for obs in recent):
            return None
        offsets = [self.lane_offset(obs) for obs in recent]
        if any(o is None for o in offsets):
            return None
        # Misma técnica que en growth_rate: la pendiente de una recta suaviza el temblor.
        # Usamos |offset| porque solo importa si se acerca al centro, no por qué lado.
        speed = statistics.linear_regression([obs.t for obs in recent], [abs(o) for o in offsets]).slope
        if speed < -MIN_LATERAL_SPEED:
            return "hacia_centro"
        if speed > MIN_LATERAL_SPEED:
            return "alejandose"
        return "estable"

    def ttc_trend(self, track_id):
        """Segundos hasta que distancia_rel llegue a 0 si sigue bajando al ritmo del último segundo.

        Complementa a ttc(): ese usa 10 frames del TAMAÑO y va por detrás cuando el de delante
        frena de golpe; la distancia, en cambio, bajaba de forma muy regular (Fase 5).
        Se lo damos a la API ya calculado: con solo "la distancia de hace 1 s" no hacía la cuenta.
        """
        hist = self.history[track_id]
        if hist[-1].cls not in GROUND_CLASSES or hist[-1].t - hist[0].t < MIN_TREND_SPAN_S:
            return None
        if any(self._truncated(obs) for obs in hist):
            return None
        # Recta sobre ~1 s de distancias (todo el historial): más estable que restar dos puntos.
        times = [obs.t for obs in hist]
        distances = [self._distance_of(obs) for obs in hist]
        drop_rate = -statistics.linear_regression(times, distances).slope  # > 0 si se acerca
        if drop_rate < MIN_DISTANCE_DROP_RATE:
            return None
        ttc = distances[-1] / drop_rate
        return ttc if ttc <= MAX_TTC_S else None

    def _truncated(self, obs):
        """¿La caja toca el borde izquierdo o derecho de la imagen? (ver EDGE_MARGIN_PX)."""
        return obs.x1 <= EDGE_MARGIN_PX or obs.x2 >= self.frame_width - EDGE_MARGIN_PX
