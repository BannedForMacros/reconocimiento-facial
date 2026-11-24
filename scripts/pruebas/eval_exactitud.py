"""
Prueba de exactitud del sistema con datos publicos.

- 100 comensales registrados en la BD real por el endpoint /api/registrar_multiframe:
  61 actores de CREMA-D (video) + 39 personas de LFW (fotos).
- 50 personas NO registradas (impostores): 30 de CREMA-D + 20 de LFW.
- Kiosco: rafagas de 5 fotogramas de OTROS videos del mismo actor (otra frase, neutral y feliz).
- Liveness: MiniFASNet en CelebA-Spoof (vivos vs ataques) + rafagas reales + foto en la mano.
- Escala: se agregan hasta 2000 personas distractoras de LFW a la galeria.

Uso (desde la raiz del proyecto, con PostgreSQL encendido):
  python scripts/pruebas/eval_exactitud.py <dir_datos> <dir_pylibs> <salida.json> <dir_cremad_IEO> <dir_celebaspoof>
    <dir_datos>/cremad2/        {actor}_DFA_NEU_XX.flv y {actor}_ITS_HAP_XX.flv (CREMA-D, GitHub CheyneyComputerScience)
    <dir_datos>/lfw/lfw_home/lfw_funneled/   LFW descomprimido
    <dir_pylibs>                carpeta con pyarrow (para leer los .parquet de CelebA-Spoof)
    <dir_cremad_IEO>            {actor}_IEO_NEU_XX.flv (video de registro)
    <dir_celebaspoof>           *.parquet del split test de CelebA-Spoof
Deja en la BD los comensales de prueba (apellido PRUEBA-CREMA / PRUEBA-LFW, DNI 9000xxxx).
Al volver a correrlo, primero borra los de la corrida anterior.
Despues: python scripts/pruebas/analizar_exactitud.py <salida.json>
"""
import sys, os, glob, io, json, time, random, base64, asyncio
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
DATOS, PYLIBS, SALIDA = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.append(PYLIBS)

import numpy as np, cv2
from loguru import logger
logger.remove(); logger.add(sys.stderr, level="WARNING")

from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from app.main import app
from app.core.config import settings
from app.core.database import AsyncSessionLocal, engine
from app.models.comensal import Comensal
from app.models.embedding import Embedding
from app.models.marcacion import Marcacion
from app.models.asistencia import Asistencia
from app.models.intento_acceso import IntentoAcceso
from app.services.cache_service import EmbeddingCache
from app.services.face_service import face_service
from app.services.liveness_service import liveness_service
from app.routes.rafaga import analizar_rafaga

rng = random.Random(2026)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

CREMA1 = sys.argv[4] if len(sys.argv) > 4 else None   # carpeta con *_IEO_NEU_XX.flv
CREMA2 = os.path.join(DATOS, "cremad2")
APELLIDO = "PRUEBA"

# ------------------------------------------------------------------ utilidades
def leer_video(p, ancho=640, alto=480):
    cap, fr = cv2.VideoCapture(p), []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.resize(f, (ancho, alto)))
    return fr

def jpg(f): return cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
def b64(f): return "data:image/jpeg;base64," + base64.b64encode(jpg(f)).decode()

def rafaga_de(frames, inicio=20, paso=5, n=5):
    """5 fotogramas separados ~170 ms (video a 30 fps), como el kiosco."""
    idx = [min(inicio + i * paso, len(frames) - 1) for i in range(n)]
    return [frames[i] for i in idx]

def foto_en_mano(frame, seed):
    """Ataque de foto impresa movida en la mano: el plano completo se mueve en bloque."""
    r = np.random.default_rng(seed); h, w = frame.shape[:2]; out = []
    for i in range(5):
        ang = r.uniform(-3, 3); s = 1 + r.uniform(-0.04, 0.04)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
        M[:, 2] += r.uniform(-12, 12, 2)
        out.append(cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return out

def variantes(img, k, seed):
    """Simula fotogramas de un video de registro a partir de una foto (pequenos movimientos/luz)."""
    r = np.random.default_rng(seed); h, w = img.shape[:2]; out = []
    for _ in range(k):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-2, 2), 1 + r.uniform(-0.03, 0.03))
        M[:, 2] += r.uniform(-5, 5, 2)
        im = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        out.append(np.clip(im.astype(np.float32) * r.uniform(0.92, 1.08), 0, 255).astype(np.uint8))
    return out

async def limpiar_pruebas():
    await engine.dispose()
    async with AsyncSessionLocal() as db:
        ids = (await db.execute(select(Comensal.id).where(Comensal.apellido.like(f"{APELLIDO}-%")))).scalars().all()
        for m in (Marcacion, Asistencia, IntentoAcceso, Embedding):
            await db.execute(delete(m).where(m.comensal_id.in_(ids)))
        await db.execute(delete(Comensal).where(Comensal.id.in_(ids)))
        await db.commit()
    await engine.dispose()
    return len(ids)

# ------------------------------------------------------------------ datos
log("cargando LFW...")
RAIZ = os.path.join(DATOS, "lfw", "lfw_home", "lfw_funneled")
nombres = sorted(d for d in os.listdir(RAIZ) if os.path.isdir(os.path.join(RAIZ, d)))
por_persona = {}
for k, nm in enumerate(nombres):
    fs = sorted(glob.glob(os.path.join(RAIZ, nm, "*.jpg")))
    if fs: por_persona[k] = fs
def lfw_img(f):
    return cv2.resize(cv2.imread(f), (500, 500))
con6 = sorted([p for p, v in por_persona.items() if len(v) >= 6]); rng.shuffle(con6)
con2 = sorted([p for p, v in por_persona.items() if 2 <= len(v) < 6]); rng.shuffle(con2)
lfw_reg_cand = con6                      # registro: 3 fotos, prueba: hasta 5 fotos distintas
lfw_desc = con2[:20]                     # desconocidos: hasta 3 fotos
usados = set(con6) | set(lfw_desc)
distractores = sorted([p for p in por_persona if p not in usados]); rng.shuffle(distractores)
distractores = distractores[:2000]
log(f"LFW: {len(por_persona)} personas, {len(con6)} con >=6 fotos")

actores = list(range(1001, 1092)); rng.shuffle(actores)
crema_reg, crema_desc = sorted(actores[:61]), sorted(actores[61:])

res = {"config": {k: getattr(settings, k) for k in (
    "MATCH_THRESHOLD", "DUPLICATE_THRESHOLD", "LIVENESS_THRESHOLD", "LIVENESS_CAPAS_ACTIVAS",
    "PROFUNDIDAD_UMBRAL", "PROFUNDIDAD_MOVIMIENTO_MIN", "DET_SIZE")}}

n = asyncio.run(limpiar_pruebas())
log(f"datos de prueba previos borrados: {n}")
EmbeddingCache._instance = None

registro = []      # {persona, fuente, dni, ok, msg, validos}
dni_seq = iter(range(90000001, 90009999))
with TestClient(app) as c:
    cache = EmbeddingCache.get_instance()
    base_previa = cache.count
    # ============================================================ 1. REGISTRO
    log("registro CREMA-D (61)...")
    for a in crema_reg:
        fr = leer_video(os.path.join(CREMA1, f"{a}_IEO_NEU_XX.flv"))
        dni = str(next(dni_seq)); t = time.time()
        r = c.post("/api/registrar_multiframe", json={"dni": dni, "nombre": f"Actor {a}",
                   "apellido": f"{APELLIDO}-CREMA", "frames": [b64(f) for f in fr[:30:2]]})
        j = r.json()
        registro.append({"persona": f"C{a}", "fuente": "crema", "dni": dni, "ok": j.get("success", False),
                         "msg": j.get("message"), "validos": j.get("frames_validos"), "seg": time.time() - t})
    log("registro LFW (39, con reintentos si falla)...")
    lfw_reg, intentos_lfw = [], 0
    for p in lfw_reg_cand:
        if len(lfw_reg) >= 39: break
        intentos_lfw += 1
        fotos = por_persona[p][:3]
        frames = [v for k, i in enumerate(fotos) for v in variantes(lfw_img(i), 5, p * 10 + k)]
        dni = str(next(dni_seq)); t = time.time()
        r = c.post("/api/registrar_multiframe", json={"dni": dni, "nombre": f"Persona {p}",
                   "apellido": f"{APELLIDO}-LFW", "frames": [b64(f) for f in frames]})
        j = r.json()
        registro.append({"persona": f"L{p}", "fuente": "lfw", "dni": dni, "ok": j.get("success", False),
                         "msg": j.get("message"), "validos": j.get("frames_validos"), "seg": time.time() - t})
        if j.get("success"): lfw_reg.append(p)
    res["registro"] = registro
    log(f"registrados: {sum(r['ok'] for r in registro)} / intentos {len(registro)}")

    # Mapa persona -> comensal_id
    dni2pers = {r["dni"]: r["persona"] for r in registro if r["ok"]}
    cid2pers = {cid: dni2pers[d["dni"]] for cid, d in cache._data.items() if d["dni"] in dni2pers}
    pers2cid = {v: k for k, v in cid2pers.items()}
    gal_ids = list(cache._ids)
    gal_mat = cache._matrix.copy()
    gal_pers = [cid2pers.get(i, f"previo{i}") for i in gal_ids]

    # ============================================================ 2. ANTI-DUPLICADO
    log("anti-duplicado: 15 registrados intentan registrarse otra vez con otro DNI...")
    dup = []
    for a in crema_reg[:15]:
        fr = leer_video(os.path.join(CREMA2, f"{a}_DFA_NEU_XX.flv"))
        r = c.post("/api/registrar_multiframe", json={"dni": str(next(dni_seq)), "nombre": f"Dup {a}",
                   "apellido": f"{APELLIDO}-DUP", "frames": [b64(f) for f in (fr * 2)[:30:2]]}).json()
        dup.append({"persona": f"C{a}", "rechazado": not r.get("success"), "msg": r.get("message")})
    res["duplicados"] = dup
    # quitar los duplicados que se hayan colado para no contaminar la galeria
    cache._data = {k: v for k, v in cache._data.items() if k in gal_ids}
    cache._ids, cache._matrix = list(gal_ids), gal_mat.copy()

    # ============================================================ 3. CONSULTAS (KIOSCO)
    def consulta_rafaga(frames, etiqueta, verdadero, extra):
        t = time.time()
        r = analizar_rafaga(frames, None, None)
        seg = time.time() - t
        q = {"tipo": "rafaga", "verdadero": verdadero, "etiqueta": etiqueta, "estado": r["estado"],
             "seg": seg, **extra}
        if r["estado"] == "OK":
            embs = np.stack([d["embedding"] for _, _, d in r["detecciones"]]).astype(np.float32)
            q["embs"] = embs
            q["validos"] = r["validos"]
            s = r["senales"]
            q["liv"] = {"minifasnet": s["minifasnet"], "prof_estado": s["profundidad"]["estado"],
                        "prof_ratio": s["profundidad"].get("ratio"),
                        "prof_mov": s["profundidad"].get("movimiento"),
                        "fondo": s["fondo"]["estado"], "es_real": r["es_real"]}
        return q

    consultas = []
    log("rafagas CREMA-D: registrados y desconocidos, 2 videos distintos cada uno...")
    for a in crema_reg + crema_desc:
        verdadero = f"C{a}" if a in crema_reg else None
        for clip in ("DFA_NEU_XX", "ITS_HAP_XX"):
            p = os.path.join(CREMA2, f"{a}_{clip}.flv")
            if not os.path.exists(p): continue
            fr = leer_video(p)
            if len(fr) < 25: continue
            consultas.append(consulta_rafaga(rafaga_de(fr), "real", verdadero,
                                             {"fuente": "crema", "clip": clip, "persona": f"C{a}"}))
    log("ataques: foto en la mano (91 actores)...")
    for a in crema_reg + crema_desc:
        fr = leer_video(os.path.join(CREMA2, f"{a}_DFA_NEU_XX.flv"))
        if len(fr) < 25: continue
        consultas.append(consulta_rafaga(foto_en_mano(fr[25], a), "foto_mano",
                                         f"C{a}" if a in crema_reg else None,
                                         {"fuente": "crema", "clip": "foto", "persona": f"C{a}"}))
    log("fotos LFW: registrados (fotos no usadas al registrar) y desconocidos...")
    def consulta_foto(img, verdadero, persona):
        t = time.time(); d = face_service.detect_face(img); seg = time.time() - t
        q = {"tipo": "foto", "verdadero": verdadero, "etiqueta": "real", "fuente": "lfw",
             "persona": persona, "seg": seg}
        if d is None or d["det_score"] < settings.DETECTION_CONFIDENCE:
            q["estado"] = "NO_FACE"; return q
        q["estado"] = "OK"; q["validos"] = 1; q["embs"] = d["embedding"][None].astype(np.float32)
        return q
    for p in lfw_reg:
        for i in por_persona[p][3:8]:
            consultas.append(consulta_foto(lfw_img(i), f"L{p}", f"L{p}"))
    for p in lfw_desc:
        for i in por_persona[p][:3]:
            consultas.append(consulta_foto(lfw_img(i), None, f"L{p}"))

    log("distractores LFW (hasta 2000 personas, 1 foto c/u)...")
    dist_embs = []
    for p in distractores:
        d = face_service.detect_face(lfw_img(por_persona[p][0]))
        if d is not None and d["det_score"] >= settings.DETECTION_CONFIDENCE:
            dist_embs.append(d["embedding"].astype(np.float32))
    dist_embs = np.stack(dist_embs)
    log(f"distractores validos: {len(dist_embs)}")

    # ============================================================ 4. LIVENESS MiniFASNet (CelebA-Spoof)
    log("MiniFASNet en CelebA-Spoof...")
    import pyarrow.parquet as pq
    celeba = []
    for f in sorted(glob.glob(os.path.join(sys.argv[5], "*.parquet"))):
        t = pq.read_table(f, columns=["Filepath", "Class"])
        for fp, cl in zip(t.column("Filepath").to_pylist(), t.column("Class").to_pylist()):
            im = cv2.imdecode(np.frombuffer(fp["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if im is None: continue
            d = face_service.detect_face(im)
            if d is None: celeba.append({"clase": cl, "score": None}); continue
            l = liveness_service.analyze(im, d["bbox"])
            celeba.append({"clase": cl, "score": l["score"], "detalle": l["detail"]})
    res["celeba"] = celeba

    # ============================================================ 5. E2E HTTP (llena asistencias)
    log("kiosco real por HTTP: 61 registrados + 30 desconocidos (video DFA)...")
    e2e = []
    for a in crema_reg + crema_desc:
        fr = leer_video(os.path.join(CREMA2, f"{a}_DFA_NEU_XX.flv"))
        if len(fr) < 25: continue
        files = [("fotos", (f"f{i}.jpg", jpg(f), "image/jpeg")) for i, f in enumerate(rafaga_de(fr))]
        t = time.time(); j = c.post("/api/reconocer_rafaga", files=files).json(); seg = time.time() - t
        nom = (j.get("comensal") or {}).get("nombre")
        e2e.append({"persona": f"C{a}", "registrado": a in crema_reg, "status": j.get("status"),
                    "message": j.get("message"), "nombre": nom, "seg": seg})
    res["e2e"] = e2e

async def borrar_dup():
    await engine.dispose()
    async with AsyncSessionLocal() as db:
        ids = (await db.execute(select(Comensal.id).where(Comensal.apellido == f"{APELLIDO}-DUP"))).scalars().all()
        for m in (Marcacion, Asistencia, IntentoAcceso, Embedding):
            await db.execute(delete(m).where(m.comensal_id.in_(ids)))
        await db.execute(delete(Comensal).where(Comensal.id.in_(ids)))
        await db.commit()
    await engine.dispose()
asyncio.run(borrar_dup())

# ------------------------------------------------------------------ guardar crudo
np.savez_compressed(SALIDA.replace(".json", ".npz"), gal_mat=gal_mat, dist_embs=dist_embs)
res["gal_pers"] = gal_pers
res["consultas"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in q.items()} for q in consultas]
res["n_registrados"] = len(gal_ids); res["base_previa"] = base_previa
json.dump(res, open(SALIDA, "w"), default=float)
log("listo ->", SALIDA)
