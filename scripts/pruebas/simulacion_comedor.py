"""
Simulacion de operacion del comedor: 100 registrados, 10 dias habiles (2 semanas), desayuno/almuerzo/cena,
30 impostores x 5 intentos, ataques (foto/pantalla) - todo por el endpoint real /api/reconocer_rafaga.

- Reloj simulado: el sistema cree que es el dia y la hora del evento (asistencias, marcaciones e intentos
  quedan con esa fecha). Cada intento queda en la tabla auditoria_pruebas con su clase real.
- Luz por horario: manana (luz natural fria), tarde (fuerte, a veces contraluz), noche (fluorescente, poca luz).
- NO se ajusta nada para obtener un numero: el resultado es el que da el sistema.

Uso (desde la raiz): python scripts/pruebas/simulacion_comedor.py [datos_pruebas] [datos_pruebas/celebaspoof] [datos_pruebas/sim_resultados.json]
Requisitos: scripts/pruebas/descargar_datos.py y scripts/pruebas/embeddings_poblacion.py. ~1 h en CPU.
"""
import sys, os, glob, json, time, random, base64, asyncio, collections, warnings, zlib
from datetime import datetime as _dt, date, timedelta
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
ARGS = sys.argv[1:] + [None] * 3
DATOS = ARGS[0] or "datos_pruebas"
CELEBA = ARGS[1] or os.path.join(DATOS, "celebaspoof")
SALIDA = ARGS[2] or os.path.join(DATOS, "sim_resultados.json")
import numpy as np, cv2
from loguru import logger; logger.remove(); logger.add(sys.stderr, level="ERROR")
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sqlalchemy import event, text
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.core.database import AsyncSessionLocal, engine
import app.routes.reconocimiento as mod_rec
import app.routes.rafaga as mod_raf
import app.services.validacion_service as mod_val
from app.models.marcacion import Marcacion
from app.models.intento_acceso import IntentoAcceso
from app.services.cache_service import EmbeddingCache
from app.services.face_service import face_service
from app.services.liveness_service import liveness_service

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
rng = random.Random(1125); nrng = np.random.default_rng(1125)
RAIZ = os.path.join(DATOS, "lfw", "lfw_home", "lfw_funneled")

# ------------------------------------------------------------------ reloj simulado
SIM = {"now": _dt(2026, 11, 9, 7, 0)}
class RelojSimulado(_dt):
    @classmethod
    def now(cls, tz=None): return SIM["now"]
mod_rec.datetime = RelojSimulado
mod_val.datetime = RelojSimulado
event.listen(Marcacion, "before_insert", lambda m, c, t: setattr(t, "fecha_hora", SIM["now"]))
event.listen(IntentoAcceso, "before_insert", lambda m, c, t: setattr(t, "fecha_hora", SIM["now"]))

# ------------------------------------------------------------------ captura de lo que decide el endpoint
CAP = {}
_analizar_original = mod_raf.analizar_rafaga
def analizar_capturado(images, colores=None, referencia=None):
    r = _analizar_original(images, colores, referencia)
    CAP["r"] = r
    if r.get("estado") == "OK":
        cache = EmbeddingCache.get_instance()
        E = np.stack([d["embedding"] for _, _, d in r["detecciones"]]).astype(np.float32)
        D = 1 - E @ cache._matrix.T
        CAP["D"] = D; CAP["ids"] = list(cache._ids)
    return r
mod_raf.analizar_rafaga = analizar_capturado

# Las personas simuladas son FOTOS: MiniFASNet las rechazaria (correcto). Para intentos de personas reales
# (registrados e impostores) se reemplaza solo la nota de MiniFASNet; el control de POCA LUZ se mantiene.
# Los ATAQUES pasan por el MiniFASNet real.
_liv_original = liveness_service.analyze
MODO = {"persona_real": True}
def liveness_sim(im, bbox):
    r = _liv_original(im, bbox)
    if MODO["persona_real"] and r.get("detail") == "ok":
        return {**r, "score": 0.97, "is_live": True}
    return r
liveness_service.analyze = liveness_sim

# ------------------------------------------------------------------ condiciones de camara por horario
def jpg(f, q=85): return cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()

def escena(img, comida, r):
    """Imagen de 640x480 como la ve la webcam del kiosco en ese horario."""
    lienzo = np.full((480, 640, 3), 0, np.uint8)
    fondo = cv2.resize(cv2.GaussianBlur(img, (0, 0), 25), (640, 480))     # fondo desenfocado del lugar
    lienzo[:] = fondo
    s = r.uniform(0.70, 0.95)                                              # distancia a la camara
    cara = cv2.resize(img, (int(500 * s * 0.9), int(500 * s * 0.9)))
    h, w = cara.shape[:2]; y0, x0 = max(0, (480 - h) // 2), max(0, (640 - w) // 2)
    lienzo[y0:y0 + min(h, 480), x0:x0 + min(w, 640)] = cara[:480, :640]
    im = lienzo.astype(np.float32)
    if comida == "desayuno":      # manana: luz natural fria, algo tenue
        im *= r.uniform(0.70, 0.95); im[..., 0] *= 1.08; ruido = r.uniform(4, 8)
    elif comida == "almuerzo":    # tarde: buena luz, a veces contraluz de la puerta
        im *= r.uniform(0.95, 1.20); ruido = r.uniform(3, 5)
        if r.random() < 0.3:
            grad = np.linspace(1.35, 0.75, 640)[None, :, None]; im *= grad
    else:                         # noche: fluorescente, poca luz, tinte, mas ruido y movimiento
        im *= r.uniform(0.40, 0.70); im[..., 2] *= 1.06; im[..., 0] *= 0.90; ruido = r.uniform(8, 14)
        if r.random() < 0.3:
            k = np.zeros((7, 7), np.float32); k[3, :] = 1 / 7; im = cv2.filter2D(im, -1, k)
    im = np.clip(im, 0, 255)
    return im, ruido

def rafaga(img, comida, r):
    """5 fotogramas de la misma persona (180 ms entre si): micro-movimiento + ruido de sensor."""
    base, ruido = escena(img, comida, r)
    out = []
    for _ in range(5):
        M = np.float32([[1, 0, r.uniform(-3, 3)], [0, 1, r.uniform(-3, 3)]])
        f = cv2.warpAffine(base, M, (640, 480), borderMode=cv2.BORDER_REFLECT)
        f = np.clip(f + nrng.normal(0, ruido, f.shape), 0, 255).astype(np.uint8)
        out.append(jpg(f, int(r.uniform(60, 85))))
    return out

def ataque_foto(img, r, tipo):
    """Foto impresa o celular sostenido en la mano frente a la camara (5 fotogramas, movimiento plano)."""
    f = img.astype(np.float32)
    if tipo == "impresa":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
        f = 0.75 * f + 0.25 * g; f = f * 0.9 + 15
    else:  # pantalla de celular: moire + brillo azulado + reflejo
        Y, X = np.mgrid[:f.shape[0], :f.shape[1]]
        f = f + 10 * np.sin(X * 1.1 + Y * 0.3)[..., None]; f[..., 0] *= 1.1
        f += 40 * np.exp(-((X - 350) ** 2 + (Y - 120) ** 2) / 6000.0)[..., None]
    f = np.clip(f, 0, 255).astype(np.uint8)
    fondo = np.full((480, 640, 3), r.randint(90, 160), np.uint8)
    out = []
    for i in range(5):
        M = cv2.getRotationMatrix2D((250, 250), r.uniform(-3, 3), r.uniform(0.62, 0.68))
        M[:, 2] += [70 + r.uniform(-10, 10), -10 + r.uniform(-8, 8)]
        fr = cv2.warpAffine(f, M, (640, 480), borderValue=(0, 0, 0))
        m = cv2.warpAffine(np.ones(f.shape[:2], np.uint8), M, (640, 480)) > 0
        im = fondo.copy(); im[m] = fr[m]
        im = np.clip(im + nrng.normal(0, 4, im.shape), 0, 255).astype(np.uint8)
        out.append(jpg(im))
    return out

# ------------------------------------------------------------------ eleccion de la muestra
def lfw_fotos(p): return sorted(glob.glob(os.path.join(RAIZ, p, "*.jpg")))
def leer(f): return cv2.resize(cv2.imread(f), (500, 500))
personas10 = sorted(d for d in os.listdir(RAIZ) if len(lfw_fotos(d)) >= 10)
F = np.load(os.path.join(DATOS, "emb_fairface.npz"))
clf = LogisticRegression(max_iter=3000, C=2.0).fit(F["Et"], F["Mt"][:, 1].astype(int))
LP = np.load(os.path.join(DATOS, "emb_lfw_primera.npz")); NLP = LP["N"].tolist()
p_lat = {p: float(clf.predict_proba(LP["E"][NLP.index(p)][None])[0, 5]) for p in personas10 if p in NLP}
# 100 registrados: personas con >=10 fotos (para tener fotos distintas en 2 semanas), priorizando rasgos latinos
registrables = sorted(p_lat, key=lambda p: -p_lat[p])
log(f"personas LFW con >=10 fotos: {len(personas10)}")

def det_score(img):
    d = face_service.detect_face(img); return d["det_score"] if d else 0

async def preparar_bd():
    await engine.dispose()
    async with AsyncSessionLocal() as db:
        await db.execute(text("""
            CREATE TABLE IF NOT EXISTS auditoria_pruebas (
              id SERIAL PRIMARY KEY, corrida VARCHAR(40), fecha_hora TIMESTAMP, dia INT, comida VARCHAR(12),
              tipo VARCHAR(12), persona VARCHAR(80), comensal_id_real INT, intento INT, condicion VARCHAR(60),
              status_sistema VARCHAR(20), mensaje TEXT, comensal_id_detectado INT, distancia FLOAT,
              clase VARCHAR(20), correcto BOOLEAN, capas_activas BOOLEAN)"""))
        await db.execute(text("DELETE FROM auditoria_pruebas WHERE corrida LIKE 'SIM-%'"))
        ids = (await db.execute(text("SELECT id FROM comensales WHERE apellido LIKE 'PRUEBA-%'"))).scalars().all()
        if ids:
            for t in ("marcaciones", "asistencias", "intentos_acceso", "embeddings"):
                await db.execute(text(f"DELETE FROM {t} WHERE comensal_id = ANY(:i)"), {"i": list(ids)})
            await db.execute(text("DELETE FROM comensales WHERE id = ANY(:i)"), {"i": list(ids)})
        await db.execute(text("DELETE FROM intentos_acceso WHERE comensal_id IS NULL"))
        await db.commit()
    await engine.dispose()
    return len(ids)

async def guardar_auditoria(filas):
    await engine.dispose()
    async with AsyncSessionLocal() as db:
        for f in filas:
            await db.execute(text("""INSERT INTO auditoria_pruebas (corrida, fecha_hora, dia, comida, tipo, persona,
                comensal_id_real, intento, condicion, status_sistema, mensaje, comensal_id_detectado, distancia, clase,
                correcto, capas_activas) VALUES (:corrida,:fecha_hora,:dia,:comida,:tipo,:persona,:comensal_id_real,
                :intento,:condicion,:status_sistema,:mensaje,:comensal_id_detectado,:distancia,:clase,:correcto,:capas_activas)"""), f)
        await db.commit()
    await engine.dispose()

log("BD: borrados comensales de prueba previos:", asyncio.run(preparar_bd()))
EmbeddingCache._instance = None
CORRIDA = "SIM-" + _dt.now().strftime("%Y%m%d-%H%M")
AUD, RES = [], {"corrida": CORRIDA}

def variantes(img, k, seed):
    r = np.random.default_rng(seed); out = []
    for _ in range(k):
        M = cv2.getRotationMatrix2D((250, 250), r.uniform(-2, 2), 1 + r.uniform(-0.03, 0.03)); M[:, 2] += r.uniform(-5, 5, 2)
        out.append(np.clip(cv2.warpAffine(img, M, (500, 500), borderMode=cv2.BORDER_REFLECT) * r.uniform(0.92, 1.08), 0, 255).astype(np.uint8))
    return out

def clasificar(tipo, cid_real, cid_det, status):
    if tipo == "registrado":
        if cid_det is None: return "NR", False
        return ("OK", True) if cid_det == cid_real else ("CF", False)
    if tipo == "impostor":
        return ("FA", False) if cid_det is not None else ("TN", True)
    return ("ATAQUE_PASA", False) if status in ("MATCH_CONFIRMED", "REJECTED", "COOLDOWN") else ("ATAQUE_BLOQUEADO", True)

def detectado(status):
    """Comensal identificado por el endpoint (misma regla de mayoria), o None si no identifico a nadie."""
    if status in ("MATCH_CONFIRMED", "REJECTED", "COOLDOWN") and "D" in CAP:
        D, ids = CAP["D"], CAP["ids"]
        j = D.argmin(1); d = D[np.arange(len(j)), j]
        votos = collections.Counter(ids[k] for k, dd in zip(j, d) if dd < settings.MATCH_THRESHOLD)
        if votos: return votos.most_common(1)[0][0], float(np.mean([dd for k, dd in zip(j, d) if ids[k] == votos.most_common(1)[0][0]]))
    return None, (float(CAP["D"].min()) if "D" in CAP else None)

def intento(c, frames, tipo, persona, cid_real, n, dia, comida, condicion, guardar_crudo):
    CAP.clear()
    files = [("fotos", (f"f{i}.jpg", b, "image/jpeg")) for i, b in enumerate(frames)]
    j = c.post("/api/reconocer_rafaga", files=files).json()
    st = j.get("status")
    cid_det, dist = detectado(st)
    clase, ok = clasificar(tipo, cid_real, cid_det, st)
    AUD.append({"corrida": CORRIDA, "fecha_hora": SIM["now"], "dia": dia, "comida": comida, "tipo": tipo,
                "persona": persona, "comensal_id_real": cid_real, "intento": n, "condicion": condicion,
                "status_sistema": st, "mensaje": (j.get("message") or "")[:200], "comensal_id_detectado": cid_det,
                "distancia": dist, "clase": clase, "correcto": ok, "capas_activas": settings.LIVENESS_CAPAS_ACTIVAS})
    if guardar_crudo and "D" in CAP:   # para el barrido de umbrales y el Rank-1
        crudo.append({"tipo": tipo, "real": cid_real, "D": CAP["D"].round(4).tolist(), "ids": CAP["ids"],
                      "comida": comida, "dia": dia, "status": st})
    return st, clase

crudo = []
with TestClient(app) as c:
    cache = EmbeddingCache.get_instance()
    # ================================================================ REGISTRO (exactamente 100)
    MODO["persona_real"] = True
    reg, fallos, dni = {}, [], 93000000
    for p in registrables:
        if len(reg) == 100: break
        fs = lfw_fotos(p)
        mejores = sorted(range(len(fs)), key=lambda k: -det_score(leer(fs[k])))[:3]   # registro cooperativo
        frames = [v for k in mejores for v in variantes(leer(fs[k]), 5, dni + k)]
        dni += 1
        j = c.post("/api/registrar_multiframe", json={"dni": str(dni), "nombre": p.replace("_", " ")[:40],
                   "apellido": "PRUEBA-SIM", "frames": ["data:image/jpeg;base64," + base64.b64encode(jpg(f, 90)).decode() for f in frames]}).json()
        if j.get("success"):
            cid = next(k for k, v in cache._data.items() if v["dni"] == str(dni))
            reg[p] = {"cid": cid, "consulta": [fs[k] for k in range(len(fs)) if k not in mejores]}
        else:
            fallos.append({"persona": p, "msg": j.get("message")})
    RES["registro"] = {"registrados": len(reg), "fallidos": len(fallos), "fallos": fallos}
    log(f"registrados: {len(reg)} | fallidos: {len(fallos)}")

    # ================================================================ IMPOSTORES: los 30 mas parecidos a la galeria
    G = cache._matrix.copy()
    cand = []
    for p in NLP:     # LFW no registrados con >=5 fotos
        if p in reg or len(lfw_fotos(p)) < 5: continue
        cand.append(("lfw", p, float((1 - G @ LP["E"][NLP.index(p)]).min())))
    Ev, Mv = F["Ev"], F["Mv"]
    for i in np.where(Mv[:, 1] == 5)[0]:   # latinos de FairFace (cada foto es otra persona)
        cand.append(("fairface", int(Mv[i, 0]), float((1 - G @ Ev[i]).min())))
    lfw_c = sorted([x for x in cand if x[0] == "lfw"], key=lambda x: x[2])[:15]
    ff_c = sorted([x for x in cand if x[0] == "fairface"], key=lambda x: x[2])[:15]
    impostores = lfw_c + ff_c
    RES["impostores"] = [{"fuente": a, "persona": str(b), "dist_al_mas_parecido": round(d, 3)} for a, b, d in impostores]
    tff = pq.read_table(os.path.join(DATOS, "fairface", "val.parquet"), columns=["image"]).column("image")
    def imp_fotos(fuente, p):
        if fuente == "lfw": return [leer(f) for f in lfw_fotos(p)[:5]]
        im = cv2.imdecode(np.frombuffer(tff[int(p)].as_py()["bytes"], np.uint8), 1)
        return [cv2.resize(im, (500, 500))] * 5      # una sola foto: 5 intentos en distintos horarios
    log("impostores elegidos (los mas parecidos):", [round(x[2], 3) for x in impostores])

    # ================================================================ CALENDARIO: 10 dias habiles x 3 comidas
    dias = [date(2026, 11, 9) + timedelta(days=k) for k in range(12) if (date(2026, 11, 9) + timedelta(days=k)).weekday() < 5][:10]
    VENTANA = {"desayuno": (6, 30, 8, 50), "almuerzo": (11, 45, 14, 45), "cena": (17, 45, 20, 15)}
    ASISTE = {"desayuno": 0.75, "almuerzo": 0.95, "cena": 0.65}
    MAX_REINTENTOS = 3
    eventos = []
    for di, d in enumerate(dias):
        for comida, (h0, m0, h1, m1) in VENTANA.items():
            for p in reg:
                if rng.random() < ASISTE[comida]:
                    t = _dt(d.year, d.month, d.day, h0, m0) + timedelta(minutes=rng.uniform(0, (h1 * 60 + m1) - (h0 * 60 + m0)))
                    eventos.append((t, di + 1, comida, "registrado", p))
    # 150 intentos de impostores: 5 por persona, en dias y horarios al azar
    for fuente, p, _ in impostores:
        for n in range(5):
            d = rng.choice(dias); comida = rng.choice(list(VENTANA)); h0, m0, h1, m1 = VENTANA[comida]
            t = _dt(d.year, d.month, d.day, h0, m0) + timedelta(minutes=rng.uniform(0, (h1 * 60 + m1) - (h0 * 60 + m0)))
            eventos.append((t, dias.index(d) + 1, comida, "impostor", (fuente, p, n)))
    eventos.sort(key=lambda e: e[0])
    log(f"eventos: {len(eventos)} ({sum(e[3]=='registrado' for e in eventos)} visitas de registrados, 150 de impostores)")

    ultimo_slot = None
    for k, (t, dia, comida, tipo, quien) in enumerate(eventos):
        if (dia, comida) != ultimo_slot:
            mod_rec._cooldowns.clear(); ultimo_slot = (dia, comida)
        r = random.Random(zlib.crc32(f"{t}{quien}".encode()))
        if tipo == "registrado":
            p = quien; info = reg[p]
            for n in range(1, MAX_REINTENTOS + 1):          # si falla, la persona vuelve a intentar
                SIM["now"] = t + timedelta(seconds=8 * (n - 1))
                img = leer(r.choice(info["consulta"]))
                st, clase = intento(c, rafaga(img, comida, r), "registrado", p, info["cid"], n, dia, comida,
                                    f"luz_{comida}", True)
                if clase in ("OK", "CF"): break
        else:
            fuente, p, n = quien
            SIM["now"] = t
            img = imp_fotos(fuente, p)[n]
            intento(c, rafaga(img, comida, r), "impostor", f"{fuente}:{p}", None, n + 1, dia, comida,
                    f"luz_{comida}", True)
        if k % 300 == 0: log(f"evento {k}/{len(eventos)} {t:%Y-%m-%d %H:%M} {comida}")

    # ================================================================ ATAQUES (MiniFASNet real)
    MODO["persona_real"] = False
    victimas = list(reg)[:20]
    spoof = []
    for f in sorted(glob.glob(os.path.join(CELEBA, "*.parquet"))):
        tb = pq.read_table(f, columns=["Filepath", "Class"])
        for fp, cl in zip(tb.column("Filepath").to_pylist(), tb.column("Class").to_pylist()):
            if cl == "spoof" and len(spoof) < 20:
                im = cv2.imdecode(np.frombuffer(fp["bytes"], np.uint8), 1)
                if im is not None and face_service.detect_face(im) is not None: spoof.append(cv2.resize(im, (500, 500)))
    RES["ataques"] = {}
    for capas in (False, True):
        settings.LIVENESS_CAPAS_ACTIVAS = capas
        cnt = collections.Counter()
        for i, v in enumerate(victimas):          # 20 suplantaciones de registrados: 10 foto impresa + 10 celular
            mod_rec._cooldowns.clear()
            SIM["now"] = _dt(2026, 11, 20, 12, 30) + timedelta(minutes=i)
            r = random.Random(500 + i)
            tipo_a = "impresa" if i % 2 == 0 else "celular"
            st, clase = intento(c, ataque_foto(leer(reg[v]["consulta"][0]), r, tipo_a), "ataque", f"{tipo_a}:{v}",
                                reg[v]["cid"], 1, 10, "almuerzo", f"ataque_{tipo_a}", False)
            cnt[(tipo_a, clase)] += 1
        for i, im in enumerate(spoof):            # 20 ataques reales fotografiados (CelebA-Spoof)
            SIM["now"] = _dt(2026, 11, 20, 13, 0) + timedelta(minutes=i)
            r = random.Random(900 + i)
            st, clase = intento(c, rafaga(im, "almuerzo", r), "ataque", f"celebaspoof:{i}", None, 1, 10, "almuerzo",
                                "ataque_celebaspoof", False)
            cnt[("celebaspoof", clase)] += 1
        RES["ataques"]["capas_activas" if capas else "modo_actual_sombra"] = {f"{a}|{b}": v for (a, b), v in cnt.items()}
        log("ataques capas", capas, dict(cnt))
    settings.LIVENESS_CAPAS_ACTIVAS = False

RES["dias"] = [str(d) for d in dias]
RES["config"] = {"MATCH_THRESHOLD": settings.MATCH_THRESHOLD, "DETECTION_CONFIDENCE": settings.DETECTION_CONFIDENCE,
                 "ASISTENCIA": ASISTE, "MAX_REINTENTOS": MAX_REINTENTOS}
RES["galeria_ids"] = {p: v["cid"] for p, v in reg.items()}
asyncio.run(guardar_auditoria(AUD))
json.dump({"res": RES, "aud": AUD, "crudo": crudo}, open(SALIDA, "w", encoding="utf-8"), default=str, ensure_ascii=False)
log("listo:", len(AUD), "intentos auditados")
