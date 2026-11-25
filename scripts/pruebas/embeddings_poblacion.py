"""Huellas ArcFace (mismo face_service del sistema) de FairFace y LFW, en cache .npz.
Genera datos_pruebas/emb_fairface.npz y emb_lfw_primera.npz (los usa la simulacion del comedor).
Uso (desde la raiz): python scripts/pruebas/embeddings_poblacion.py [datos_pruebas]   (~25 min en CPU)"""
import sys, os, glob, random, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "."); DATOS = sys.argv[1] if len(sys.argv) > 1 else "datos_pruebas"
import numpy as np, cv2
from loguru import logger; logger.remove()
import pyarrow.parquet as pq
from app.services.face_service import face_service
import time; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
rng = random.Random(7)

def emb(img):
    d = face_service.detect_face(img)
    if d is None: return None
    return d["embedding"].astype(np.float32), d["det_score"], min(d["face_width"], d["face_height"])

def fairface(nombre, elegir):
    t = pq.read_table(os.path.join(DATOS, "fairface", nombre), columns=["race", "age", "gender"])
    race, age, gen = (np.array(t.column(c).to_pylist()) for c in ("race", "age", "gender"))
    idx = elegir(race, age)
    t = pq.read_table(os.path.join(DATOS, "fairface", nombre), columns=["image"]).column("image")
    E, M = [], []
    for k, i in enumerate(idx):
        r = emb(cv2.imdecode(np.frombuffer(t[int(i)].as_py()["bytes"], np.uint8), 1))
        if r is None: continue
        E.append(r[0]); M.append((int(i), int(race[i]), int(age[i]), int(gen[i]), r[1], r[2]))
        if k % 500 == 0: log(nombre, k, "/", len(idx))
    return np.stack(E), np.array(M, dtype=np.float32)

ADULTO = lambda a: (a >= 2) & (a <= 6)          # 10-19 ... 50-59 anios
def val_sel(race, age):
    lat = np.where((race == 5) & ADULTO(age))[0].tolist()
    otros = []
    for c in (0, 1, 2, 3, 4, 6):
        ids = np.where((race == c) & ADULTO(age))[0].tolist(); rng.shuffle(ids); otros += ids[:150]
    return lat + otros
def train_sel(race, age):
    out = []
    for c in range(7):
        ids = np.where((race == c) & ADULTO(age))[0].tolist(); rng.shuffle(ids); out += ids[:450]
    return out

out = os.path.join(DATOS, "emb_fairface.npz")
if not os.path.exists(out):
    Ev, Mv = fairface("val.parquet", val_sel)
    Et, Mt = fairface("train0.parquet", train_sel)
    np.savez_compressed(out, Ev=Ev, Mv=Mv, Et=Et, Mt=Mt); log("FairFace listo", Ev.shape, Et.shape)

RAIZ = os.path.join(DATOS, "lfw", "lfw_home", "lfw_funneled")
out = os.path.join(DATOS, "emb_lfw_primera.npz")
if not os.path.exists(out):
    E, N = [], []
    personas = sorted(d for d in os.listdir(RAIZ) if len(glob.glob(os.path.join(RAIZ, d, "*.jpg"))) >= 3)
    for k, p in enumerate(personas):
        f = sorted(glob.glob(os.path.join(RAIZ, p, "*.jpg")))[0]
        r = emb(cv2.resize(cv2.imread(f), (500, 500)))
        if r is None: continue
        E.append(r[0]); N.append(p)
        if k % 300 == 0: log("LFW", k, "/", len(personas))
    np.savez_compressed(out, E=np.stack(E), N=np.array(N)); log("LFW listo", len(N))
