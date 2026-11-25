"""
Pruebas A (lentes) y B (poblacion latina y parecidos), con 100 registrados EXACTOS por galeria,
registrados por el endpoint real /api/registrar_multiframe (con liveness), en la BD real.

Uso: python scripts/pruebas/prueba_lentes_latinos.py <dir_datos> <dir_pylibs_o_""> <salida.json>
Requiere vgg_latinos.pkl (scripts/pruebas/seleccion_vggface2.py) y las huellas de embeddings_poblacion.py.
"""
import sys, os, glob, json, time, random, base64, asyncio, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "."); DATOS, LIBS, SALIDA = sys.argv[1], sys.argv[2], sys.argv[3]; sys.path.append(LIBS)
import numpy as np, cv2
from loguru import logger; logger.remove(); logger.add(sys.stderr, level="ERROR")
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
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

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
rng = random.Random(2026)
RAIZ = os.path.join(DATOS, "lfw", "lfw_home", "lfw_funneled")
THR = settings.MATCH_THRESHOLD
RES = {"config": {"MATCH_THRESHOLD": THR, "DETECTION_CONFIDENCE": settings.DETECTION_CONFIDENCE,
                  "MIN_FACE_SIZE": settings.MIN_FACE_SIZE, "DUPLICATE_THRESHOLD": settings.DUPLICATE_THRESHOLD}}

# ------------------------------------------------------------------ utilidades
def lfw_path(persona, num): return os.path.join(RAIZ, persona, f"{persona}_{num:04d}.jpg")
def leer(p): return cv2.resize(cv2.imread(p), (500, 500))
def jpg(f, q=90): return cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
def b64(f): return "data:image/jpeg;base64," + base64.b64encode(jpg(f)).decode()

def webcam_barata(img, seed):
    """Webcam USB barata en el comedor: menos resolucion, poca luz, ruido de sensor y JPEG fuerte."""
    r = np.random.default_rng(seed); h, w = img.shape[:2]
    s = r.uniform(0.35, 0.55)                                   # rostro de ~90-140 px
    im = cv2.resize(cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), (w, h))
    g = r.uniform(0.55, 0.95)                                   # luz baja/irregular
    im = np.clip(255 * (im / 255.0) ** (1 / g) * r.uniform(0.6, 1.0), 0, 255)
    im = np.clip(im + r.normal(0, r.uniform(4, 10), im.shape), 0, 255).astype(np.uint8)
    return cv2.imdecode(np.frombuffer(jpg(im, int(r.uniform(35, 65))), np.uint8), 1)

def variantes(img, k, seed):
    r = np.random.default_rng(seed); h, w = img.shape[:2]; out = []
    for _ in range(k):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-2, 2), 1 + r.uniform(-0.03, 0.03))
        M[:, 2] += r.uniform(-5, 5, 2)
        im = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        out.append(np.clip(im.astype(np.float32) * r.uniform(0.92, 1.08), 0, 255).astype(np.uint8))
    return out

def analizar(img):
    """Misma calidad que el kiosco: deteccion >= DETECTION_CONFIDENCE y rostro >= MIN_FACE_SIZE."""
    d = face_service.detect_face(img)
    if d is None: return {"estado": "NO_FACE"}
    q = {"det": float(d["det_score"]), "emb": d["embedding"].astype(np.float32),
         "tam": float(min(d["face_width"], d["face_height"]))}
    q["estado"] = "OK" if q["det"] >= settings.DETECTION_CONFIDENCE and q["tam"] >= settings.MIN_FACE_SIZE else "CALIDAD"
    return q

async def limpiar(prefijo="PRUEBA-"):
    await engine.dispose()
    async with AsyncSessionLocal() as db:
        ids = (await db.execute(select(Comensal.id).where(Comensal.apellido.like(f"{prefijo}%")))).scalars().all()
        for m in (Marcacion, Asistencia, IntentoAcceso, Embedding):
            await db.execute(delete(m).where(m.comensal_id.in_(ids)))
        await db.execute(delete(Comensal).where(Comensal.id.in_(ids)))
        await db.commit()
    await engine.dispose()
    return len(ids)

# ------------------------------------------------------------------ atributos de lentes (LFW, Columbia)
L = open(os.path.join(DATOS, "lfw_attributes.txt")).read().splitlines()
H = L[1].lstrip("#").strip().split("\t")
lentes = collections.defaultdict(lambda: {"sin": [], "lentes": [], "oscuros": []})
for l in L[2:]:
    r = dict(zip(H, l.split("\t")))
    ne, eg, sg = float(r["No Eyewear"]), float(r["Eyeglasses"]), float(r["Sunglasses"])
    k = "oscuros" if (sg > 0.5 and sg > eg) else "lentes" if (eg > 0.5 and ne < 0) else "sin" if (ne > 0.5 and eg < 0 and sg < 0) else None
    p = r["person"].replace(" ", "_")
    if k and os.path.exists(lfw_path(p, int(r["imagenum"]))): lentes[p][k].append(int(r["imagenum"]))

# ------------------------------------------------------------------ clasificador de poblacion (solo para ELEGIR la muestra)
F = np.load(os.path.join(DATOS, "emb_fairface.npz"))
Ev, Mv, Et, Mt = F["Ev"], F["Mv"], F["Et"], F["Mt"]
clf = LogisticRegression(max_iter=3000, C=2.0).fit(Et, Mt[:, 1].astype(int))
pv = clf.predict(Ev); yv = Mv[:, 1].astype(int)
RES["selector_latino"] = {"exactitud_7_grupos_%": round(100 * float((pv == yv).mean()), 1),
                          "recall_latino_%": round(100 * float((pv[yv == 5] == 5).mean()), 1),
                          "precision_latino_%": round(100 * float((yv[pv == 5] == 5).mean()), 1)}
log("selector:", RES["selector_latino"])
LP = np.load(os.path.join(DATOS, "emb_lfw_primera.npz"))
p_lat = dict(zip(LP["N"].tolist(), clf.predict_proba(LP["E"])[:, 5].tolist()))
fotos = {p: sorted(glob.glob(os.path.join(RAIZ, p, "*.jpg"))) for p in p_lat}

# Impostores latinos "en la naturaleza" (FairFace, adultos 10-59): cada foto es una persona distinta
imp_lat = Ev[Mv[:, 1] == 5]; imp_lat_meta = Mv[Mv[:, 1] == 5]
imp_otros = Ev[Mv[:, 1] != 5]
log(f"impostores FairFace: {len(imp_lat)} latinos, {len(imp_otros)} de otros grupos")

# ------------------------------------------------------------------ registro de una galeria (exactamente 100)
from app.services.liveness_service import liveness_service
_liv_original = liveness_service.analyze

def registrar_galeria(c, candidatos, fotos_reg, etiqueta, dni0):
    # Las imagenes de los datasets SON fotos: MiniFASNet las rechaza (correcto). Para medir IDENTIDAD se
    # desactiva el liveness solo aqui; deteccion, calidad y anti-duplicado siguen siendo los del endpoint.
    liveness_service.analyze = lambda im, bbox: {"score": 1.0, "is_live": True, "detail": "ok"}
    try:
        return _registrar(c, candidatos, fotos_reg, etiqueta, dni0)
    finally:
        liveness_service.analyze = _liv_original

def _registrar(c, candidatos, fotos_reg, etiqueta, dni0):
    reg, fallos, dni = [], [], dni0
    for p in candidatos:
        if len(reg) == 100: break
        fs = fotos_reg(p)
        if not fs: continue
        per = max(1, 15 // len(fs)); frames = []
        for k, f in enumerate(fs): frames += variantes(leer(f) if isinstance(f, str) else f, per, sum(map(ord, p)) * 7 + k)
        frames = (frames * 2)[:15]
        dni += 1
        j = c.post("/api/registrar_multiframe", json={"dni": str(dni), "nombre": str(p).replace("_", " ")[:40],
                   "apellido": f"PRUEBA-{etiqueta}", "frames": [b64(f) for f in frames]}).json()
        (reg if j.get("success") else fallos).append({"persona": p, "dni": str(dni), "msg": j.get("message"),
                                                     "usadas": [os.path.basename(f) if isinstance(f, str) else k for k, f in enumerate(fs)]})
    cache = EmbeddingCache.get_instance()
    d2p = {r["dni"]: r["persona"] for r in reg}
    pers = [d2p.get(cache._data[i]["dni"], f"otro{i}") for i in cache._ids]
    return reg, fallos, cache._matrix.copy(), pers

def decidir(q, G, P, thr=THR):
    if q["estado"] != "OK": return "CALIDAD", None, None
    D = 1 - G @ q["emb"]; j = int(D.argmin())
    return ("MATCH" if D[j] < thr else "NO_MATCH"), P[j], float(D[j])

def resumen(consultas, G, P, thr=THR):
    c = collections.Counter()
    for q in consultas:
        est, quien, d = decidir(q, G, P, thr)
        if q["verdadero"]:
            c["n"] += 1
            c[{"CALIDAD": "calidad", "NO_MATCH": "no_rec"}.get(est, "ok" if quien == q["verdadero"] else "otro")] += 1
        else:
            c["imp"] += 1; c["acepta"] += est == "MATCH"
    p = lambda a, b: round(100 * a / b, 2) if b else None
    return {"reconoce_%": p(c["ok"], c["n"]), "confunde_%": p(c["otro"], c["n"]), "no_reconoce_%": p(c["no_rec"], c["n"]),
            "calidad_%": p(c["calidad"], c["n"]), "n": c["n"],
            "acepta_desconocido_%": p(c["acepta"], c["imp"]), "n_imp": c["imp"]}

def barrido(consultas, G, P):
    return {str(t): resumen(consultas, G, P, t) for t in (0.35, 0.40, 0.45, 0.50, 0.55)}

def con_distancias(consultas, G, P):
    """Distancia a la propia plantilla (si esta registrado) para cada consulta OK."""
    out = []
    for q in consultas:
        if q["estado"] != "OK" or not q["verdadero"] or q["verdadero"] not in P: continue
        out.append(float(1 - G[P.index(q["verdadero"])] @ q["emb"]))
    return out

def stats(v):
    v = np.asarray(v)
    return {"n": int(len(v)), "media": round(float(v.mean()), 3), "p50": round(float(np.median(v)), 3),
            "p95": round(float(np.percentile(v, 95)), 3), "max": round(float(v.max()), 3),
            "sobre_0.45_%": round(100 * float((v >= 0.45).mean()), 2)} if len(v) else {}

def impostores_vs(G, P, E_imp, umbrales=(0.40, 0.45, 0.50, 0.55)):
    """Cada impostor contra los 100: el mas parecido. Devuelve FAR y los 'dobles' mas cercanos."""
    D = 1 - E_imp @ G.T
    mn = D.min(1); arg = D.argmin(1)
    por_reg = D.min(0)                                     # el doble mas parecido de cada registrado
    return {"n": int(len(E_imp)),
            "acepta_%": {str(t): round(100 * float((mn < t).mean()), 3) for t in umbrales},
            "mas_parecido_media": round(float(mn.mean()), 3), "mas_parecido_min": round(float(mn.min()), 3),
            "p1": round(float(np.percentile(mn, 1)), 3),
            "registrados_con_doble_bajo_umbral": {str(t): int((por_reg < t).sum()) for t in umbrales},
            "doble_por_registrado": {"media": round(float(por_reg.mean()), 3), "min": round(float(por_reg.min()), 3),
                                     "p5": round(float(np.percentile(por_reg, 5)), 3)},
            "top10": sorted([(P[int(arg[i])], round(float(mn[i]), 3)) for i in np.argsort(mn)[:10]], key=lambda x: x[1])}

def degradar(q_img, seed):
    return analizar(webcam_barata(q_img, seed))

n = asyncio.run(limpiar()); log("comensales PRUEBA-* previos borrados:", n)

# ======================================================================== PRUEBA A: LENTES
cand_A = sorted([p for p, v in lentes.items() if v["sin"] and v["lentes"]], key=lambda p: -len(lentes[p]["lentes"]))
# Respaldo: criterio mas amplio (puntaje de lentes > 0 en vez de > 0.5) por si alguno falla en el registro
for l in L[2:]:
    r = dict(zip(H, l.split("	"))); pp = r["person"].replace(" ", "_")
    if float(r["Eyeglasses"]) > 0 and float(r["Sunglasses"]) < 0 and lentes[pp]["sin"] and pp not in cand_A:
        lentes[pp].setdefault("lentes_amplio", []).append(int(r["imagenum"]))
for pp, v in list(lentes.items()):
    if v.get("lentes_amplio") and pp not in cand_A:
        v["lentes"] = v["lentes"] or v["lentes_amplio"]; cand_A.append(pp)
log(f"Prueba A: {len(cand_A)} candidatos con fotos con y sin lentes")
EmbeddingCache._instance = None
with TestClient(app) as c:
    regA, fallA, GA, PA = registrar_galeria(c, cand_A, lambda p: [lfw_path(p, n) for n in lentes[p]["sin"][:3]], "LENTES", 91000000)
log(f"A registrados: {len(regA)} | fallidos: {len(fallA)}")
usadasA = {r["persona"]: set(r["usadas"]) for r in regA}
qA = {"sin_lentes": [], "con_lentes": [], "oscuros": []}
seed = 0
for r in regA:
    p = r["persona"]
    for clase, nums in (("sin_lentes", lentes[p]["sin"]), ("con_lentes", lentes[p]["lentes"]), ("oscuros", lentes[p]["oscuros"])):
        for num in nums[:6]:
            f = lfw_path(p, num)
            if os.path.basename(f) in usadasA[p]: continue
            img = leer(f); seed += 1
            q = analizar(img); q["verdadero"] = p; q["deg"] = False; qA[clase].append(q)
            q2 = degradar(img, seed); q2["verdadero"] = p; q2["deg"] = True; qA[clase].append(q2)
# Lentes oscuros de personas que no estan en la galeria A: plantilla offline con sus fotos sin lentes
extra_osc, GA2, PA2 = [], GA.copy(), list(PA)
for p, v in lentes.items():
    if p in usadasA or not (v["sin"] and v["oscuros"]): continue
    embs = [analizar(leer(lfw_path(p, n))) for n in v["sin"][:3]]
    embs = [e["emb"] for e in embs if e["estado"] == "OK"]
    if not embs: continue
    t = np.mean(embs, 0); GA2 = np.vstack([GA2, t / np.linalg.norm(t)]); PA2.append(p)
    for num in v["oscuros"][:6]:
        img = leer(lfw_path(p, num)); seed += 1
        q = analizar(img); q["verdadero"] = p; q["deg"] = False; extra_osc.append(q)
        q2 = degradar(img, seed); q2["verdadero"] = p; q2["deg"] = True; extra_osc.append(q2)
osc_all = qA["oscuros"] + extra_osc
RES["A"] = {
    "registrados": len(regA), "fallidos_registro": len(fallA),
    "fallos_msgs": collections.Counter((f["msg"] or "")[:80] for f in fallA).most_common(4),
    "personas_oscuros": len({q["verdadero"] for q in osc_all}),
    "resultados": {f"{clase}{'_webcam' if deg else ''}": resumen([q for q in qs if q["deg"] == deg], GA2 if clase == "oscuros" else GA,
                                                                 PA2 if clase == "oscuros" else PA)
                   for clase, qs in (("sin_lentes", qA["sin_lentes"]), ("con_lentes", qA["con_lentes"]), ("oscuros", osc_all))
                   for deg in (False, True)},
    "barrido_con_lentes_webcam": barrido([q for q in qA["con_lentes"] if q["deg"]], GA, PA),
    "barrido_oscuros": barrido([q for q in osc_all if not q["deg"]], GA2, PA2),
    "distancias": {f"{clase}{'_webcam' if deg else ''}": stats(con_distancias([q for q in qs if q["deg"] == deg],
                                                                              GA2 if clase == "oscuros" else GA, PA2 if clase == "oscuros" else PA))
                   for clase, qs in (("sin_lentes", qA["sin_lentes"]), ("con_lentes", qA["con_lentes"]), ("oscuros", osc_all))
                   for deg in (False, True)},
    "impostores_latinos": impostores_vs(GA, PA, imp_lat),
}
# Inverso: registrado CON lentes, consultado SIN lentes (plantilla offline con la misma regla de promedio)
GI, PI_, qI = [], [], []
for r in regA:
    p = r["persona"]
    e = [analizar(leer(lfw_path(p, n))) for n in lentes[p]["lentes"][:3]]
    e = [x["emb"] for x in e if x["estado"] == "OK"]
    if not e: continue
    t = np.mean(e, 0); GI.append(t / np.linalg.norm(t)); PI_.append(p)
GI = np.stack(GI)
for r in regA:
    p = r["persona"]
    if p not in PI_: continue
    for num in lentes[p]["sin"][3:9]:
        img = leer(lfw_path(p, num)); seed += 1
        q = analizar(img); q["verdadero"] = p; q["deg"] = False; qI.append(q)
        q2 = degradar(img, seed); q2["verdadero"] = p; q2["deg"] = True; qI.append(q2)
RES["A"]["inverso_registro_con_lentes"] = {"foto": resumen([q for q in qI if not q["deg"]], GI, PI_),
                                          "webcam": resumen([q for q in qI if q["deg"]], GI, PI_), "galeria": len(PI_)}
log("A listo:", json.dumps(RES["A"]["resultados"])[:600])
json.dump(RES, open(SALIDA, "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
asyncio.run(limpiar("PRUEBA-LENTES"))

# ======================================================================== PRUEBA B: POBLACION LATINA Y PARECIDOS
import pickle
V = pickle.load(open(os.path.join(DATOS, "vgg_latinos.pkl"), "rb"))
# Los recortes de VGGFace2 son muy ajustados: margen gris para que el detector vea la cabeza completa
dec = lambda b: cv2.copyMakeBorder(cv2.imdecode(np.frombuffer(b, np.uint8), 1), 112, 112, 112, 112, cv2.BORDER_CONSTANT, value=(128, 128, 128))
vfotos = {f"V{l}": [dec(b) for b in V["datos"][l]["fotos"]] for l in V["orden"]}
vprob = {f"V{l}": V["datos"][l]["prob"] for l in V["orden"]}
cand_B = list(vfotos)
# Registro cooperativo: las 3 fotos mas nitidas/frontales (mayor det_score); las demas quedan para consultar
mejores = {}
def fotos_registro(p):
    if p not in mejores:
        sc = [(analizar(im).get("det", 0), k) for k, im in enumerate(vfotos[p])]
        mejores[p] = [k for _, k in sorted(sc, reverse=True)[:3]]
    return [vfotos[p][k] for k in mejores[p]]
log(f"Prueba B: {len(cand_B)} candidatos latinos de VGGFace2 (prob media del top100 {np.mean([vprob[p] for p in cand_B[:100]]):.2f})")
EmbeddingCache._instance = None
with TestClient(app) as c:
    regB, fallB, GB, PB = registrar_galeria(c, cand_B, fotos_registro, "LATINO", 92000000)
log(f"B registrados: {len(regB)} | fallidos: {len(fallB)}")
regset = {r["persona"] for r in regB}
qB = []
for r in regB:
    p = r["persona"]
    for k, img in enumerate(vfotos[p]):
        if k in mejores[p]: continue
        if sum(1 for q in qB if q["verdadero"] == p and not q["deg"]) >= 8: break
        seed += 1
        q = analizar(img); q["verdadero"] = p; q["deg"] = False; qB.append(q)
        q2 = degradar(img, seed); q2["verdadero"] = p; q2["deg"] = True; qB.append(q2)
# Desconocidos latinos de VGGFace2 (candidatos que no entraron a la galeria): misma poblacion, fotos en la naturaleza
imp_lfw = []
for p in [p for p in cand_B if p not in regset and p not in {f["persona"] for f in fallB}]:
    for img in vfotos[p][:3]:
        seed += 1
        q = analizar(img); q["verdadero"] = None; q["deg"] = False; imp_lfw.append(q)
        q2 = degradar(img, seed); q2["verdadero"] = None; q2["deg"] = True; imp_lfw.append(q2)
RES["B_desconocidos_vgg_personas"] = len([p for p in cand_B if p not in regset and p not in {f["persona"] for f in fallB}])
# Impostores FairFace degradados como webcam (muestra de 600)
t = pq.read_table(os.path.join(DATOS, "fairface", "val.parquet"), columns=["image"]).column("image")
idx_lat = imp_lat_meta[:, 0].astype(int).tolist(); rng.shuffle(idx_lat)
imp_ff_web = []
for i in idx_lat[:600]:
    img = cv2.imdecode(np.frombuffer(t[i].as_py()["bytes"], np.uint8), 1); seed += 1
    q = degradar(img, seed)
    if q["estado"] == "OK": imp_ff_web.append(q["emb"])
imp_ff_web = np.stack(imp_ff_web)
# Pares dentro de la galeria (registrados que se parecen entre si)
DG = 1 - GB @ GB.T; np.fill_diagonal(DG, 9)
pares = sorted({tuple(sorted((PB[i], PB[j]))): round(float(DG[i, j]), 3)
                for i, j in zip(*np.where(DG < 0.75))}.items(), key=lambda x: x[1])[:10]
RES["B"] = {
    "registrados": len(regB), "fallidos_registro": len(fallB),
    "fallos_msgs": collections.Counter((f["msg"] or "")[:80] for f in fallB).most_common(4),
    "prob_latino_media_galeria": round(float(np.mean([vprob[r["persona"]] for r in regB])), 3),
    "personas": [r["persona"] for r in regB],
    "genuinos_foto": resumen([q for q in qB if not q["deg"]] + [q for q in imp_lfw if not q["deg"]], GB, PB),
    "genuinos_webcam": resumen([q for q in qB if q["deg"]] + [q for q in imp_lfw if q["deg"]], GB, PB),
    "barrido_foto": barrido([q for q in qB if not q["deg"]] + [q for q in imp_lfw if not q["deg"]], GB, PB),
    "barrido_webcam": barrido([q for q in qB if q["deg"]] + [q for q in imp_lfw if q["deg"]], GB, PB),
    "distancias_foto": stats(con_distancias([q for q in qB if not q["deg"]], GB, PB)),
    "distancias_webcam": stats(con_distancias([q for q in qB if q["deg"]], GB, PB)),
    "impostores_latinos_fairface": impostores_vs(GB, PB, imp_lat),
    "impostores_latinos_fairface_webcam": impostores_vs(GB, PB, imp_ff_web),
    "impostores_otros_grupos_fairface": impostores_vs(GB, PB, imp_otros),
    "pares_mas_parecidos_en_galeria": pares,
    "min_entre_registrados": round(float(DG.min()), 3),
}
# Calidad relajada: cuantas consultas que fallaron por calidad se reconocerian con DETECTION_CONFIDENCE=0.70
def relajado(qs, G, P):
    r = [q for q in qs if q["estado"] == "CALIDAD" and q["det"] >= 0.70 and q["tam"] >= settings.MIN_FACE_SIZE]
    ok = sum(decidir({**q, "estado": "OK"}, G, P)[1] == q["verdadero"] and decidir({**q, "estado": "OK"}, G, P)[0] == "MATCH" for q in r)
    return {"rescatables_con_0.70": len(r), "de_ellos_bien_reconocidos": int(ok),
            "fallos_calidad_total": sum(q["estado"] in ("CALIDAD", "NO_FACE") for q in qs)}
RES["B"]["calidad_0.70_webcam"] = relajado([q for q in qB if q["deg"]], GB, PB)
RES["B"]["calidad_0.70_foto"] = relajado([q for q in qB if not q["deg"]], GB, PB)
log("B listo")
json.dump(RES, open(SALIDA, "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
log("guardado en", SALIDA)
