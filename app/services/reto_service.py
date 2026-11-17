"""
Prueba de vida ACTIVA (reto aleatorio de giro de cabeza).

Se pide girar la cabeza a un lado elegido al azar y volver al frente. El giro se mide
con los 5 puntos de SCRFD: desplazamiento de la nariz respecto al centro de los ojos,
dividido por la distancia entre ojos. En una foto o pantalla (superficie plana) ese
valor casi no cambia aunque se gire el celular; en una cabeza real si.
Un video pregrabado no sabe que lado se pedira.
"""
import random
import time

import numpy as np

from app.core.config import settings

LADOS = ("izquierda", "derecha")


def medir_giro(kps: np.ndarray) -> float:
    ojo_i, ojo_d, nariz = kps[0], kps[1], kps[2]
    dist = float(np.linalg.norm(ojo_d - ojo_i)) + 1e-6
    return float((nariz[0] - (ojo_i[0] + ojo_d[0]) / 2) / dist)


class RetoService:

    def __init__(self):
        self._estado = {}  # session_key -> dict

    def _nuevo(self, session: str) -> dict:
        e = {"lado": random.choice(LADOS), "inicio": time.time(), "fase": "pendiente",
             "cumplido_ts": 0.0, "max_giro": 0.0}
        self._estado[session] = e
        return e

    def actualizar(self, session: str, kps: np.ndarray) -> dict:
        """Registra el cuadro actual y devuelve si el reto esta cumplido (y vigente)."""
        ahora = time.time()
        e = self._estado.get(session)
        if e is None or (e["fase"] != "cumplido" and ahora - e["inicio"] > settings.RETO_TIMEOUT_S) \
                or (e["fase"] == "cumplido" and ahora - e["cumplido_ts"] > settings.RETO_VALIDEZ_S):
            e = self._nuevo(session)

        giro = medir_giro(kps)
        # signo: + = la nariz va hacia la derecha de la IMAGEN (izquierda de la persona)
        signo = settings.RETO_SIGNO_IZQUIERDA if e["lado"] == "izquierda" else -settings.RETO_SIGNO_IZQUIERDA
        hacia_lado = giro * signo
        e["max_giro"] = max(e["max_giro"], hacia_lado)

        if e["fase"] == "pendiente" and hacia_lado >= settings.RETO_GIRO_MIN:
            e["fase"] = "girado"
        elif e["fase"] == "girado" and abs(giro) <= settings.RETO_FRONTAL_MAX:
            e["fase"] = "cumplido"
            e["cumplido_ts"] = ahora

        instruccion = {
            "pendiente": f"Gira la cabeza a tu {e['lado'].upper()}",
            "girado": "Bien, ahora mira de frente",
            "cumplido": "Verificando identidad...",
        }[e["fase"]]
        return {"cumplido": e["fase"] == "cumplido", "fase": e["fase"], "lado": e["lado"],
                "giro": round(giro, 3), "instruccion": instruccion}

    def consumir(self, session: str):
        """Tras confirmar la asistencia, el proximo comensal recibe un reto nuevo."""
        self._estado.pop(session, None)


reto_service = RetoService()
