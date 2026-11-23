"""
Liveness multi-fotograma basado en fisica (solo camara web).

Capa PROFUNDIDAD (paralaje):
    Un rostro real es un objeto 3D: cuando la cabeza se mueve (aunque sea un
    poco) o la cara gesticula, sus 106 puntos faciales NO se pueden explicar con
    una sola transformacion de plano (homografia). Una foto impresa o una foto en
    un celular es PLANA: todos sus puntos se mueven exactamente segun una
    homografia, sin importar como se mueva la mano.
    -> residuo de la homografia alto = 3D (real); residuo ~ ruido = plano (ataque).
    Si casi no hubo movimiento, la prueba es "inconclusa" (no rechaza).

Capa LUZ (desafio de color con la pantalla):
    La pantalla del kiosco muestra una secuencia de colores al azar. La piel
    real, a 40-60 cm, refleja esa luz: su cromaticidad sigue la secuencia. La
    cara mostrada en un celular/tablet emite su propia luz y casi no la sigue.
    Un video pregrabado no puede conocer la secuencia (cambia en cada intento).
    -> correlacion entre colores mostrados y cambio de color del rostro.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.config import settings


# ====================================================================== #
#  PROFUNDIDAD (paralaje / no-planaridad)                                 #
# ====================================================================== #

def _escala(p: np.ndarray) -> float:
    """Tamano del rostro: raiz de la distancia cuadratica media al centroide."""
    c = p - p.mean(axis=0)
    return float(np.sqrt((c ** 2).sum(axis=1).mean()))


def analizar_profundidad(landmarks: list[np.ndarray]) -> dict:
    """
    Args:
        landmarks: lista de arreglos (106, 2), uno por fotograma, en orden.

    Returns:
        dict con estado ('real' | 'plano' | 'inconcluso'), no_planaridad
        (residuo normalizado, x1000) y movimiento (normalizado, x1000).
    """
    pts = [np.asarray(l, dtype=np.float64) for l in landmarks if l is not None]
    if len(pts) < 3:
        return {"estado": "inconcluso", "motivo": "pocos_fotogramas",
                "no_planaridad": 0.0, "movimiento": 0.0}

    escala = np.mean([_escala(p) for p in pts])
    if escala < 1e-6:
        return {"estado": "inconcluso", "motivo": "escala_invalida",
                "no_planaridad": 0.0, "movimiento": 0.0}

    residuos, movimientos = [], []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pts[i].astype(np.float32), pts[j].astype(np.float32)
            # Movimiento no explicable por traslacion pura (rotacion, escala,
            # perspectiva o deformacion): es lo que "pone a prueba" la planaridad.
            d = (b - b.mean(0)) - (a - a.mean(0))
            movimientos.append(float(np.sqrt((d ** 2).sum(1).mean())) / escala)

            H, _ = cv2.findHomography(a, b, 0)  # minimos cuadrados con los 106 puntos
            if H is None:
                continue
            proy = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2)
            residuos.append(float(np.sqrt(((proy - b) ** 2).sum(1).mean())) / escala)

    if not residuos:
        return {"estado": "inconcluso", "motivo": "homografia_fallida",
                "no_planaridad": 0.0, "movimiento": 0.0}

    # Percentil alto: el par de fotogramas con mas evidencia, robusto a 1 outlier
    no_plan = float(np.percentile(residuos, 90)) * 1000
    mov = float(np.percentile(movimientos, 90)) * 1000

    # El error del detector de puntos crece con el movimiento (aun en una foto),
    # por eso se compara el residuo RELATIVO al movimiento, no el absoluto.
    # Medido con CREMA-D: real ~0.67, foto en mano ~0.15, video en pantalla ~0.45.
    ratio = no_plan / mov if mov > 1e-6 else 0.0

    # Con una foto totalmente quieta solo queda ruido (movimiento <= ~8): el
    # ratio no significa nada y la prueba no puede decidir.
    if mov < settings.PROFUNDIDAD_MOVIMIENTO_MIN:
        estado, motivo = "inconcluso", "sin_movimiento"
    elif ratio >= settings.PROFUNDIDAD_UMBRAL:
        estado, motivo = "real", "3d"
    else:
        estado, motivo = "plano", "movimiento_planar"

    return {"estado": estado, "motivo": motivo, "ratio": round(ratio, 3),
            "no_planaridad": round(no_plan, 2), "movimiento": round(mov, 2)}


# ====================================================================== #
#  FONDO QUIETO (camara fija del kiosco)                                  #
# ====================================================================== #
#
# En el kiosco la camara esta FIJA: el fondo real detras de la persona
# (pared, comedor) no se mueve. Un celular/tablet/foto lo sostiene una mano,
# y ninguna mano esta quieta: todo lo que rodea la cara dentro del aparato
# (fondo grabado + borde del equipo) se desplaza EN BLOQUE, como una lamina.
# Esto delata los videos, que la prueba de profundidad no puede frenar.

def _mascara_anillo(shape, bbox, ancho: float = 1.8) -> np.ndarray:
    """Franja alrededor de la cabeza (costados y arriba), sin cara, pelo ni hombros."""
    H, W = shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    m = np.zeros((H, W), np.uint8)
    # zona exterior: hasta 'ancho' anchos de cara a cada lado y ~0.55*ancho altos hacia arriba
    ox1, ox2 = int(max(0, x1 - ancho * w)), int(min(W, x2 + ancho * w))
    oy1, oy2 = int(max(0, y1 - 0.55 * ancho * h)), int(min(H, y2))  # hasta la barbilla: sin hombros
    m[oy1:oy2, ox1:ox2] = 255
    # excluir cabeza con margen amplio (pelo largo, orejas, gorra)
    ix1, ix2 = int(max(0, x1 - 0.5 * w)), int(min(W, x2 + 0.5 * w))
    iy1 = int(max(0, y1 - 0.6 * h))
    m[iy1:oy2, ix1:ix2] = 0
    return m


def _puntos_con_textura(gray: np.ndarray, mask: np.ndarray):
    """Esquinas con textura REAL (umbral absoluto): una pared lisa no aporta puntos."""
    p = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.001,
                                minDistance=5, mask=mask)
    if p is None:
        return None
    eig = cv2.cornerMinEigenVal(gray.astype(np.float32) / 255.0, blockSize=5)
    xy = p.reshape(-1, 2).astype(int)
    fuertes = eig[xy[:, 1], xy[:, 0]] >= settings.FONDO_TEXTURA_MIN
    return p[fuertes] if fuertes.any() else None


def analizar_fondo(frames: list[np.ndarray], bboxes: list[list]) -> dict:
    """
    Returns:
        dict con estado ('quieto' | 'se_mueve' | 'inconcluso'), desplazamiento
        mediano del anillo (px), coherencia (fraccion de puntos que se mueven
        juntos como una lamina) y numero de puntos con textura.
    """
    if len(frames) < 3:
        return {"estado": "inconcluso", "motivo": "pocos_fotogramas",
                "desplazamiento": 0.0, "coherencia": 0.0, "puntos": 0}

    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    mask = _mascara_anillo(g0.shape, bboxes[0])
    p0 = _puntos_con_textura(g0, mask)
    n_pts = 0 if p0 is None else len(p0)
    if n_pts < settings.FONDO_MIN_PUNTOS:
        return {"estado": "inconcluso", "motivo": "fondo_sin_textura",
                "lamina": 0, "fraccion": 0.0, "puntos": n_pts}

    mejor_n, mejor_frac = 0, 0.0
    cx1, cx2 = bboxes[0][0], bboxes[0][2]
    for f in frames[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(g0, g, p0, None, winSize=(21, 21), maxLevel=3)
        # verificacion ida-vuelta: descarta seguimientos poco confiables
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(g, g0, p1, None, winSize=(21, 21), maxLevel=3)
        fb = np.linalg.norm(pb.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
        ok = (st.reshape(-1) == 1) & (stb.reshape(-1) == 1) & (fb < 0.5)
        if ok.sum() < settings.FONDO_MIN_PUNTOS:
            continue
        a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
        moving = np.linalg.norm(b - a, axis=1) > settings.FONDO_DESPLAZAMIENTO_MIN
        # Lamina: el grupo mas grande de puntos que se mueve segun UNA homografia
        if moving.sum() >= settings.FONDO_LAMINA_MIN:
            _, inl = cv2.findHomography(a[moving], b[moving], cv2.RANSAC, 1.0)
            if inl is None:
                continue
            sel = a[moving][inl.reshape(-1) == 1]
            # Un aparato RODEA la cara: su lamina aparece a ambos lados de la cabeza.
            # Una persona en la fila esta a un solo lado (y se mueve por su cuenta).
            izq, der = int((sel[:, 0] < cx1).sum()), int((sel[:, 0] > cx2).sum())
            if min(izq, der) < settings.FONDO_LAMINA_POR_LADO:
                continue
            n = len(sel)
            if n > mejor_n:
                mejor_n, mejor_frac = n, n / len(a)

    if mejor_n >= settings.FONDO_LAMINA_MIN and mejor_frac >= settings.FONDO_COHERENCIA_MIN:
        estado, motivo = "se_mueve", "lamina_rigida"
    else:
        estado, motivo = "quieto", "fondo_fijo"
    return {"estado": estado, "motivo": motivo, "lamina": mejor_n,
            "fraccion": round(mejor_frac, 3), "puntos": n_pts}


# ====================================================================== #
#  FONDO DE REFERENCIA (el kiosco conoce su propio fondo)                 #
# ====================================================================== #
#
# Cuando no hay nadie, el kiosco memoriza su fondo vacio (y lo actualiza
# poco a poco: luz del dia, noche). Alrededor de una persona real se ve ESE
# mismo fondo. Un celular o tablet -en la mano o en un soporte- lo tapa con su
# bisel y con el fondo del video, grabado en otro lugar.

def _gris_suave(im: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(np.float32)


class ReferenciaFondo:
    """Modelo del fondo vacio por kiosco (promedio movil lento)."""

    def __init__(self):
        self._ref: dict[str, np.ndarray] = {}
        self._n: dict[str, int] = {}

    def actualizar(self, kiosco: str, frame: np.ndarray):
        g = _gris_suave(frame)
        ref = self._ref.get(kiosco)
        if ref is None or ref.shape != g.shape:
            self._ref[kiosco], self._n[kiosco] = g, 1
            return
        a = settings.FONDO_REF_APRENDIZAJE
        self._ref[kiosco] = (1 - a) * ref + a * g
        self._n[kiosco] += 1

    def obtener(self, kiosco: str) -> np.ndarray | None:
        if self._n.get(kiosco, 0) < settings.FONDO_REF_MIN_MUESTRAS:
            return None
        return self._ref[kiosco]


referencia_fondo = ReferenciaFondo()


def comparar_con_referencia(frames: list[np.ndarray], bboxes: list[list],
                            referencia: np.ndarray | None) -> dict:
    """
    Returns:
        dict con estado ('coincide' | 'no_coincide' | 'sin_referencia') y
        fraccion (0-1) del anillo alrededor de la cabeza que coincide con el
        fondo memorizado.
    """
    if referencia is None:
        return {"estado": "sin_referencia", "fraccion": None, "bloques": 0}
    B = settings.FONDO_REF_BLOQUE
    fr, nb = [], 0
    for im, bb in zip(frames, bboxes):
        g = _gris_suave(im)
        if g.shape != referencia.shape:
            return {"estado": "sin_referencia", "fraccion": None, "bloques": 0}
        m = _mascara_anillo(g.shape, bb, settings.FONDO_REF_ANCHO) > 0
        H, W = g.shape
        coinc = info = 0
        # Correlacion local (ZNCC) por bloques: compara TEXTURA, no brillo.
        # Inmune a cambios de exposicion, y un fondo liso (video) no puede
        # "parecerse" a un fondo con textura aunque tenga el mismo gris.
        for y in range(0, H - B + 1, B):
            for x in range(0, W - B + 1, B):
                if m[y:y + B, x:x + B].mean() < 0.9:
                    continue
                r = referencia[y:y + B, x:x + B]
                if r.std() < settings.FONDO_REF_TEXTURA_MIN:
                    continue                      # bloque liso en la referencia: no informa
                info += 1
                c = g[y:y + B, x:x + B]
                if c.std() < 1e-3:
                    continue
                z = float(((r - r.mean()) * (c - c.mean())).mean() / (r.std() * c.std()))
                coinc += z >= settings.FONDO_REF_ZNCC_MIN
        if info >= settings.FONDO_REF_MIN_BLOQUES:
            fr.append(coinc / info)
            nb = max(nb, info)
    if not fr:
        # El fondo del kiosco es liso alrededor de la cabeza: no hay con que comparar.
        return {"estado": "sin_referencia", "fraccion": None, "bloques": nb}
    f = float(np.median(fr))
    estado = "coincide" if f >= settings.FONDO_REF_MIN else "no_coincide"
    return {"estado": estado, "fraccion": round(f, 3), "bloques": nb}


# ====================================================================== #
#  LUZ (desafio de color con la pantalla)                                 #
# ====================================================================== #

# Paleta suave (evita rojo saturado, recomendacion WCAG 2.3.1 para fotosensibles)
PALETA = {
    "blanco": (255, 255, 255),
    "azul": (60, 100, 255),
    "verde": (70, 255, 110),
    "ambar": (255, 175, 50),
    "magenta": (230, 90, 230),
}


@dataclass
class Desafio:
    id: str
    secuencia: list[str]
    creado: float = field(default_factory=time.time)


class GestorDesafios:
    """Emite desafios de un solo uso con secuencia aleatoria (anti-video grabado)."""

    def __init__(self, ttl: float = 30.0):
        self.ttl = ttl
        self._activos: dict[str, Desafio] = {}

    def crear(self) -> Desafio:
        self._limpiar()
        nombres = list(PALETA)
        k = settings.LUZ_NUM_COLORES
        # Secuencia aleatoria sin repetir colores consecutivos
        sec = [secrets.choice(nombres)]
        while len(sec) < k:
            c = secrets.choice(nombres)
            if c != sec[-1]:
                sec.append(c)
        d = Desafio(id=secrets.token_urlsafe(16), secuencia=sec)
        self._activos[d.id] = d
        return d

    def consumir(self, desafio_id: str) -> Desafio | None:
        self._limpiar()
        return self._activos.pop(desafio_id, None)

    def _limpiar(self):
        ahora = time.time()
        for k in [k for k, d in self._activos.items() if ahora - d.creado > self.ttl]:
            del self._activos[k]


gestor_desafios = GestorDesafios()


def _lineal(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255) a luz lineal aproximada (gamma 2.2)."""
    return (np.clip(rgb, 0, 255) / 255.0) ** 2.2


def _roi_piel(bbox: list) -> tuple[slice, slice]:
    """Zona central del rostro (mejillas, nariz, frente baja): mayormente piel."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (slice(int(y1 + 0.30 * h), int(y1 + 0.80 * h)),
            slice(int(x1 + 0.22 * w), int(x1 + 0.78 * w)))


def _roi_fondo(bbox: list, shape) -> tuple[slice, slice] | None:
    """Franja del fondo junto al rostro (referencia de luz ambiental lejana)."""
    H, W = shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    xs = int(max(0, x1 - 0.9 * w)), int(max(0, x1 - 0.3 * w))
    if xs[1] - xs[0] < 10:  # sin espacio a la izquierda: usar la derecha
        xs = int(min(W, x2 + 0.3 * w)), int(min(W, x2 + 0.9 * w))
    if xs[1] - xs[0] < 10:
        return None
    return slice(int(max(0, y1)), int(min(H, y2))), slice(*xs)


def _cromaticidad(lin: np.ndarray) -> np.ndarray:
    s = lin.sum(axis=-1, keepdims=True) + 1e-6
    return lin / s


def analizar_luz(frames: list[np.ndarray], bboxes: list[list], colores: list[str]) -> dict:
    """
    Args:
        frames: fotogramas BGR, cada uno capturado mientras la pantalla mostraba
                colores[i].
        bboxes: bbox del rostro en cada fotograma.
        colores: nombre del color de PALETA mostrado en cada fotograma.

    Returns:
        dict con estado ('real' | 'falla' | 'inconcluso'), correlacion (-1..1)
        y respuesta (magnitud relativa del cambio de color en la piel).
    """
    if len(frames) < 3 or len(set(colores)) < 3:
        return {"estado": "inconcluso", "motivo": "pocos_colores",
                "correlacion": 0.0, "respuesta": 0.0}

    obs, esp, fondo = [], [], []
    for im, bb, col in zip(frames, bboxes, colores):
        ys, xs = _roi_piel(bb)
        roi = im[ys, xs]
        if roi.size == 0:
            continue
        bgr = roi.reshape(-1, 3).mean(0)
        obs.append(_cromaticidad(_lineal(bgr[::-1])))           # RGB
        esp.append(_cromaticidad(_lineal(np.array(PALETA[col], dtype=np.float64))))
        rf = _roi_fondo(bb, im.shape)
        if rf is not None and im[rf].size:
            fondo.append(_cromaticidad(_lineal(im[rf].reshape(-1, 3).mean(0)[::-1])))

    if len(obs) < 3:
        return {"estado": "inconcluso", "motivo": "roi_invalida",
                "correlacion": 0.0, "respuesta": 0.0}

    obs, esp = np.array(obs), np.array(esp)
    d_obs = (obs - obs.mean(0)).ravel()
    d_esp = (esp - esp.mean(0)).ravel()
    if d_obs.std() < 1e-9 or d_esp.std() < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(d_obs, d_esp)[0, 1])

    # Magnitud: cuanto se movio la cromaticidad de la piel respecto a lo mostrado
    respuesta = float(np.linalg.norm(d_obs) / (np.linalg.norm(d_esp) + 1e-9))

    resp_fondo = None
    if len(fondo) == len(obs):
        f = np.array(fondo)
        resp_fondo = float(np.linalg.norm((f - f.mean(0)).ravel()) / (np.linalg.norm(d_esp) + 1e-9))

    if corr >= settings.LUZ_CORRELACION_MIN and respuesta >= settings.LUZ_RESPUESTA_MIN:
        estado, motivo = "real", "sigue_la_luz"
    else:
        estado, motivo = "falla", "no_sigue_la_luz"

    return {"estado": estado, "motivo": motivo, "correlacion": round(corr, 3),
            "respuesta": round(respuesta, 4),
            "respuesta_fondo": None if resp_fondo is None else round(resp_fondo, 4)}
