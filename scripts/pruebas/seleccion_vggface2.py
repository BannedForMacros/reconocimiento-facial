"""Etapa 2: elegir identidades latinas de VGGFace2 (promedio del selector en 4 fotos por persona)
y guardar hasta 14 fotos de las 400 mas probables. Uso: python scripts/pruebas/seleccion_vggface2.py <dir_datos> <dir_pylibs_o_"">
(Necesita 10 partes de chronopt-research/cropped-vggface2-224 en <dir_datos>/vgg/, ~4.8 GB)"""
import sys, os, glob, random, pickle, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "."); DATOS, LIBS = sys.argv[1], sys.argv[2]; sys.path.append(LIBS)
import numpy as np, cv2
from loguru import logger; logger.remove()
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from app.services.face_service import face_service
T0 = time.time(); rng = random.Random(11)
def pad(im): return cv2.copyMakeBorder(im, 112, 112, 112, 112, cv2.BORDER_CONSTANT, value=(128, 128, 128))
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

F = np.load(os.path.join(DATOS, "emb_fairface.npz"))
clf = LogisticRegression(max_iter=3000, C=2.0).fit(F["Et"], F["Mt"][:, 1].astype(int))
pv, yv = clf.predict(F["Ev"]), F["Mv"][:, 1].astype(int)
log("selector en FairFace val: exactitud %.1f%% | recall latino %.1f%% | precision latino %.1f%%" % (
    100 * (pv == yv).mean(), 100 * (pv[yv == 5] == 5).mean(), 100 * (yv[pv == 5] == 5).mean()))

prob, idx_por_id, archivo_de = {}, {}, {}
for f in sorted(glob.glob(os.path.join(DATOS, "vgg", "*.parquet")), key=lambda x: int(os.path.basename(x)[:-8])):
    lab = np.array(pq.read_table(f, columns=["label"]).column("label").to_pylist())
    img = pq.read_table(f, columns=["image"]).column("image")
    for l in np.unique(lab):
        ii = np.where(lab == l)[0].tolist(); rng.shuffle(ii)
        E = []
        for i in ii[:4]:
            d = face_service.detect_face(pad(cv2.imdecode(np.frombuffer(img[i].as_py()["bytes"], np.uint8), 1)))
            if d is not None: E.append(d["embedding"])
        if len(E) >= 2:
            prob[int(l)] = float(clf.predict_proba(np.stack(E))[:, 5].mean())
            idx_por_id[int(l)], archivo_de[int(l)] = ii, f
    log(os.path.basename(f), "identidades acumuladas:", len(prob))

orden = sorted(prob, key=lambda l: -prob[l])
log("prob>=0.5:", sum(p >= 0.5 for p in prob.values()), "| >=0.4:", sum(p >= 0.4 for p in prob.values()),
    "| top100 min:", round(prob[orden[99]], 3), "| top400 min:", round(prob[orden[min(399, len(orden)-1)]], 3))
sel = orden[:400]
datos = {}
for f in set(archivo_de[l] for l in sel):
    img = pq.read_table(f, columns=["image"]).column("image")
    for l in [l for l in sel if archivo_de[l] == f]:
        datos[l] = {"prob": prob[l], "fotos": [img[i].as_py()["bytes"] for i in idx_por_id[l][:14]]}
pickle.dump({"orden": sel, "datos": datos, "prob_todos": prob}, open(os.path.join(DATOS, "vgg_latinos.pkl"), "wb"))
# hoja de contacto de las 120 primeras para revision visual
th = [cv2.resize(cv2.imdecode(np.frombuffer(datos[l]["fotos"][0], np.uint8), 1), (80, 80)) for l in sel[:120]]
filas = [np.hstack(th[i:i + 15]) for i in range(0, 120, 15)]
cv2.imwrite(os.path.join(DATOS, "contacto_latinos.jpg"), np.vstack(filas))
log("listo")
