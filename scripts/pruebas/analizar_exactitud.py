"""Calcula todas las metricas a partir de la salida de eval_exactitud.py (.json + .npz).
Uso: python scripts/pruebas/analizar_exactitud.py <salida.json>"""
import sys, json, math, collections
import numpy as np

R = json.load(open(sys.argv[1]))
Z = np.load(sys.argv[1].replace(".json", ".npz"))
GAL, DIST = Z["gal_mat"], Z["dist_embs"]
PERS = R["gal_pers"]
OUT = {}
pct = lambda a, b: round(100 * a / b, 1) if b else None

# ---------------- registro
reg = R["registro"]
for f in ("crema", "lfw"):
    rr = [r for r in reg if r["fuente"] == f]
    OUT[f"registro_{f}"] = {"intentos": len(rr), "ok": sum(r["ok"] for r in rr),
                            "seg_medio": round(np.mean([r["seg"] for r in rr]), 2),
                            "fallos": collections.Counter((r["msg"] or "")[:90] for r in rr if not r["ok"]).most_common(5)}
OUT["duplicados"] = {"intentos": len(R["duplicados"]), "rechazados": sum(d["rechazado"] for d in R["duplicados"]),
                     "msgs": collections.Counter((d["msg"] or "")[:60] for d in R["duplicados"]).most_common(3)}

# ---------------- identidad
def decidir(q, thr, gal, pers):
    """Misma regla del endpoint: cada fotograma vota por su vecino mas cercano si d<thr; gana con >=60%."""
    if q["estado"] != "OK": return "CALIDAD", None, None
    E = np.asarray(q["embs"], np.float32)
    D = 1 - E @ gal.T
    j = D.argmin(1); d = D[np.arange(len(j)), j]
    votos = collections.Counter(pers[k] for k, dd in zip(j, d) if dd < thr)
    nec = math.ceil(q["validos"] * 0.6)
    if not votos or votos.most_common(1)[0][1] < nec: return "NO_MATCH", None, float(d.min())
    w = votos.most_common(1)[0][0]
    return "MATCH", w, float(np.mean([dd for k, dd in zip(j, d) if pers[k] == w and dd < thr]))

def evaluar(qs, thr, gal, pers):
    c = collections.Counter()
    for q in qs:
        est, quien, _ = decidir(q, thr, gal, pers)
        if q["verdadero"]:
            c["gen"] += 1
            if est == "CALIDAD": c["gen_calidad"] += 1
            elif est == "NO_MATCH": c["gen_rechazo"] += 1
            elif quien == q["verdadero"]: c["gen_ok"] += 1
            else: c["gen_otro"] += 1
        else:
            c["imp"] += 1
            if est == "MATCH": c["imp_acepta"] += 1
    return {"TAR_%": pct(c["gen_ok"], c["gen"]), "confunde_con_otro_%": pct(c["gen_otro"], c["gen"]),
            "no_reconocido_%": pct(c["gen_rechazo"], c["gen"]), "falla_calidad_%": pct(c["gen_calidad"], c["gen"]),
            "FAR_desconocido_%": pct(c["imp_acepta"], c["imp"]), "n_gen": c["gen"], "n_imp": c["imp"]}

Q = R["consultas"]
for q in Q:
    if q["verdadero"] and q["verdadero"] not in PERS: q["verdadero"] = None
sets = {
    "kiosco_crema_rafaga": [q for q in Q if q["fuente"] == "crema" and q["etiqueta"] == "real"],
    "kiosco_crema_neutral": [q for q in Q if q["fuente"] == "crema" and q.get("clip") == "DFA_NEU_XX"],
    "kiosco_crema_feliz": [q for q in Q if q["fuente"] == "crema" and q.get("clip") == "ITS_HAP_XX"],
    "lfw_foto": [q for q in Q if q["fuente"] == "lfw"],
}
UMB = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
OUT["identidad"] = {n: {str(t): evaluar(qs, t, GAL, PERS) for t in UMB} for n, qs in sets.items()}

# escala: galeria con distractores
escala = {}
for extra in (0, 400, 900, 1900):
    extra = min(extra, len(DIST))
    gal = np.vstack([GAL, DIST[:extra]]) if extra else GAL
    pers = PERS + [f"D{i}" for i in range(extra)]
    escala[str(len(GAL) + extra)] = {s: evaluar(sets[s], 0.45, gal, pers) for s in ("kiosco_crema_rafaga", "lfw_foto")}
OUT["escala_umbral_0.45"] = escala

# distancias
def dist_stats(qs):
    gen, imp = [], []
    for q in qs:
        if q["estado"] != "OK": continue
        E = np.asarray(q["embs"], np.float32); D = 1 - E @ GAL.T
        if q["verdadero"]:
            k = PERS.index(q["verdadero"]); gen.append(float(D[:, k].mean()))
            Dm = np.delete(D, k, 1); imp.append(float(Dm.min(1).mean()))
        else:
            imp.append(float(D.min(1).mean()))
    f = lambda v: {"n": len(v), "media": round(np.mean(v), 3), "p5": round(np.percentile(v, 5), 3),
                   "p95": round(np.percentile(v, 95), 3), "min": round(min(v), 3), "max": round(max(v), 3)}
    return {"mismo_rostro": f(gen), "impostor_mas_cercano": f(imp)}
OUT["distancias"] = {n: dist_stats(qs) for n, qs in sets.items() if n in ("kiosco_crema_rafaga", "lfw_foto")}

# pares de impostores globales (distractores vs galeria) -> probabilidad por comparacion
Dimp = 1 - DIST @ GAL.T
OUT["pares_impostor"] = {str(t): {"p_por_par": float((Dimp < t).mean()),
                                  "galeria_1000_%": round(100 * (1 - (1 - float((Dimp < t).mean())) ** 1000), 2),
                                  "galeria_5000_%": round(100 * (1 - (1 - float((Dimp < t).mean())) ** 5000), 2)}
                         for t in UMB}
OUT["pares_impostor_n"] = int(Dimp.size)

# ---------------- liveness
cel = [c for c in R["celeba"] if c["score"] is not None and c.get("detalle") == "ok"]
viv = np.array([c["score"] for c in cel if c["clase"] == "live"]); ata = np.array([c["score"] for c in cel if c["clase"] == "spoof"])
reales = np.array([q["liv"]["minifasnet"] for q in Q if q.get("liv") and q["etiqueta"] == "real" and q["fuente"] == "crema"])
OUT["minifasnet"] = {"n_vivos_celeba": len(viv), "n_ataques_celeba": len(ata), "n_rafagas_reales_crema": len(reales),
                     "sin_rostro_celeba": sum(c["score"] is None for c in R["celeba"]),
                     "barrido": {str(t): {"ataques_pasan_APCER_%": pct((ata >= t).sum(), len(ata)),
                                          "vivos_rechazados_celeba_BPCER_%": pct((viv < t).sum(), len(viv)),
                                          "rafagas_reales_rechazadas_crema_%": pct((reales < t).sum(), len(reales))}
                                 for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}}
try:
    from sklearn.metrics import roc_auc_score
    OUT["minifasnet"]["AUC"] = round(roc_auc_score(np.r_[np.ones(len(viv)), np.zeros(len(ata))], np.r_[viv, ata]), 3)
except Exception: pass

def capas(etq):
    qs = [q for q in Q if q.get("liv") and q["etiqueta"] == etq]
    return {"n": len(qs), "profundidad": dict(collections.Counter(q["liv"]["prof_estado"] for q in qs)),
            "fondo_lamina": dict(collections.Counter(q["liv"]["fondo"] for q in qs)),
            "minifasnet_media": round(float(np.mean([q["liv"]["minifasnet"] for q in qs])), 3),
            "pasa_modo_actual_%": pct(sum(q["liv"]["es_real"] for q in qs), len(qs)),
            "pasa_profundidad_activa_%": pct(sum(q["liv"]["minifasnet"] >= 0.5 and q["liv"]["prof_estado"] != "plano" for q in qs), len(qs))}
OUT["capas"] = {"rafaga_real": capas("real"), "foto_en_mano": capas("foto_mano")}

# ---------------- tiempos y e2e
OUT["tiempo_rafaga_seg"] = round(float(np.median([q["seg"] for q in Q if q["tipo"] == "rafaga"])), 2)
e2e = R["e2e"]
OUT["e2e"] = {"registrados": dict(collections.Counter(e["status"] for e in e2e if e["registrado"])),
              "desconocidos": dict(collections.Counter(e["status"] for e in e2e if not e["registrado"])),
              "seg_mediana": round(float(np.median([e["seg"] for e in e2e])), 2),
              "mensajes": collections.Counter(e["message"][:70] for e in e2e if e["message"]).most_common(6)}
json.dump(OUT, open(sys.argv[1].replace(".json", "_metricas.json"), "w"), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
