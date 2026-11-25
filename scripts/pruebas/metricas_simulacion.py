"""Metricas de la simulacion del comedor.
Uso: python scripts/pruebas/metricas_simulacion.py [datos_pruebas/sim_resultados.json]"""
import sys, json, math, collections
import numpy as np
from scipy import stats as st

if len(sys.argv) < 2: sys.argv.append("datos_pruebas/sim_resultados.json")
J = json.load(open(sys.argv[1], encoding="utf-8"))
RES, AUD, CR = J["res"], J["aud"], J["crudo"]
OUT = {"corrida": RES["corrida"], "dias": RES["dias"], "registro": {k: v for k, v in RES["registro"].items() if k != "fallos"}}

def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * (c - m) / d, 2), round(100 * (c + m) / d, 2))
def cp_sup(k, n):   # Clopper-Pearson superior al 95% (para k=0 equivale a ~3/n)
    return round(100 * st.beta.ppf(0.975, k + 1, n - k), 3) if k < n else 100.0
def pct(k, n): return round(100 * k / n, 2) if n else None

def metricas(filas):
    c = collections.Counter(f["clase"] for f in filas)
    TP, FN, CONF, FP, TN = c["OK"], c["NR"], c["CF"], c["FA"], c["TN"]
    gen, imp = TP + FN + CONF, FP + TN
    prec = TP / (TP + FP + CONF) if TP + FP + CONF else 0
    rec = TP / gen if gen else 0
    return {"intentos_registrados": gen, "intentos_impostores": imp,
            "OK": TP, "NR": FN, "CF": CONF, "FA": FP, "TN": TN,
            "accuracy_%": pct(TP + TN, gen + imp), "accuracy_IC95": wilson(TP + TN, gen + imp),
            "precision_%": round(100 * prec, 2), "recall_TAR_%": round(100 * rec, 2), "recall_IC95": wilson(TP, gen),
            "F1_%": round(100 * 2 * prec * rec / (prec + rec), 2) if prec + rec else 0,
            "FRR_falsos_negativos_%": pct(FN + CONF, gen), "FRR_IC95": wilson(FN + CONF, gen),
            "FAR_falsos_positivos_%": pct(FP, imp), "FAR_limite_sup_95_%": cp_sup(FP, imp),
            "confusion_%": pct(CONF, gen), "confusion_limite_sup_95_%": cp_sup(CONF, gen),
            "especificidad_%": pct(TN, imp)}

per = [f for f in AUD if f["tipo"] in ("registrado", "impostor")]
OUT["global_por_intento"] = metricas(per)
OUT["por_comida"] = {m: metricas([f for f in per if f["comida"] == m]) for m in ("desayuno", "almuerzo", "cena")}
OUT["por_dia"] = {d: metricas([f for f in per if f["dia"] == d])["recall_TAR_%"] for d in sorted({f["dia"] for f in per})}
OUT["motivos_FN"] = dict(collections.Counter(f["status_sistema"] for f in per if f["clase"] == "NR"))
OUT["por_numero_de_intento"] = {n: metricas([f for f in per if f["tipo"] == "registrado" and f["intento"] == n])["recall_TAR_%"] for n in (1, 2, 3)}

# Por visita: la persona llega y reintenta hasta 3 veces
visitas = collections.defaultdict(list)
for f in AUD:
    if f["tipo"] == "registrado": visitas[(f["dia"], f["comida"], f["persona"])].append(f)
v_ok1 = sum(v[0]["clase"] == "OK" for v in visitas.values())
v_ok = sum(any(x["clase"] == "OK" for x in v) for v in visitas.values())
v_conf = sum(any(x["clase"] == "CF" for x in v) for v in visitas.values())
OUT["por_visita"] = {"visitas": len(visitas), "pasa_al_primer_intento_%": pct(v_ok1, len(visitas)),
                     "pasa_en_hasta_3_intentos_%": pct(v_ok, len(visitas)), "IC95": wilson(v_ok, len(visitas)),
                     "se_va_sin_pasar": len(visitas) - v_ok - v_conf, "confundido_con_otro": v_conf,
                     "intentos_promedio": round(np.mean([len(v) for v in visitas.values()]), 2),
                     "por_comida_%": {m: pct(sum(any(x["clase"] == "OK" for x in v) for k, v in visitas.items() if k[1] == m),
                                            sum(1 for k in visitas if k[1] == m)) for m in ("desayuno", "almuerzo", "cena")}}
# Impostores: por persona (5 intentos)
imp = collections.defaultdict(list)
for f in AUD:
    if f["tipo"] == "impostor": imp[f["persona"]].append(f)
OUT["impostores"] = {"personas": len(imp), "intentos": sum(len(v) for v in imp.values()),
                     "personas_que_entraron_alguna_vez": sum(any(x["clase"] == "FA" for x in v) for v in imp.values()),
                     "intentos_aceptados": sum(x["clase"] == "FA" for v in imp.values() for x in v),
                     "distancia_media_al_registrado_mas_parecido": None}

# ---------------- Rank-1, distancias y barrido de umbral (solo intentos que pasaron calidad)
def voto(D, ids, thr):
    j = D.argmin(1); d = D[np.arange(len(j)), j]
    v = collections.Counter(ids[k] for k, dd in zip(j, d) if dd < thr)
    nec = math.ceil(len(j) * 0.6)
    if not v or v.most_common(1)[0][1] < nec: return None
    return v.most_common(1)[0][0]
gen = [c for c in CR if c["tipo"] == "registrado"]; impc = [c for c in CR if c["tipo"] == "impostor"]
r1 = sum(voto(np.array(c["D"]), c["ids"], 9.0) == c["real"] for c in gen)
OUT["rank1_R1_%"] = pct(r1, len(gen)); OUT["rank1_n"] = len(gen)
dg = [float(np.mean(np.array(c["D"])[:, c["ids"].index(c["real"])])) for c in gen]
di = [float(np.array(c["D"]).min(1).mean()) for c in impc]
OUT["distancias"] = {"mismo_rostro": {"media": round(np.mean(dg), 3), "p95": round(np.percentile(dg, 95), 3), "p99": round(np.percentile(dg, 99), 3), "max": round(max(dg), 3)},
                     "impostor_mas_parecido": {"media": round(np.mean(di), 3), "p1": round(np.percentile(di, 1), 3), "min": round(min(di), 3)}}
OUT["impostores"]["distancia_media_al_registrado_mas_parecido"] = OUT["distancias"]["impostor_mas_parecido"]["media"]
# Cola estimada (ajuste normal de las distancias de impostores): probabilidad de bajar del umbral
mu, sd = np.mean(di), np.std(di)
OUT["FAR_estimado_por_cola_normal"] = {str(t): f"{st.norm.cdf((t - mu) / sd):.2e}" for t in (0.45, 0.50, 0.55, 0.60)}
fallos_calidad = sum(1 for f in per if f["tipo"] == "registrado" and f["clase"] == "NR" and f["status_sistema"] != "NO_MATCH")
n_gen_total = OUT["global_por_intento"]["intentos_registrados"]
barr = {}
for t in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
    tp = sum(voto(np.array(c["D"]), c["ids"], t) == c["real"] for c in gen)
    conf = sum((lambda x: x is not None and x != c["real"])(voto(np.array(c["D"]), c["ids"], t)) for c in gen)
    fp = sum(voto(np.array(c["D"]), c["ids"], t) is not None for c in impc)
    tn = len(impc) - fp
    prec = tp / (tp + fp + conf) if tp + fp + conf else 0; rec = tp / n_gen_total
    barr[str(t)] = {"recall_%": round(100 * rec, 2), "FAR_%": pct(fp, len(impc)), "FAR_lim_sup_%": cp_sup(fp, len(impc)),
                    "confusiones": conf, "precision_%": round(100 * prec, 2),
                    "F1_%": round(200 * prec * rec / (prec + rec), 2) if prec + rec else 0,
                    "accuracy_%": pct(tp + tn, n_gen_total + len(impc))}
OUT["barrido_umbral"] = barr
ata = collections.Counter(("capas_activas" if f["capas_activas"] else "modo_actual", f["condicion"], f["clase"]) for f in AUD if f["tipo"] == "ataque")
OUT["ataques_APCER"] = {}
for modo in ("modo_actual", "capas_activas"):
    for cond in sorted({k[1] for k in ata}):
        pasa, blo = ata[(modo, cond, "ATAQUE_PASA")], ata[(modo, cond, "ATAQUE_BLOQUEADO")]
        if pasa + blo: OUT["ataques_APCER"][f"{modo}|{cond}"] = {"pasan": pasa, "n": pasa + blo, "APCER_%": pct(pasa, pasa + blo),
                                                                  "IC95": wilson(pasa, pasa + blo), "limite_sup_95_%": cp_sup(pasa, pasa + blo)}

# Q de Cochran por turno: bloques alumno-dia que fueron a los 3 turnos; exito = reconocido en el primer intento
blo = collections.defaultdict(dict)
for f in AUD:
    if f["tipo"] == "registrado" and f["intento"] == 1: blo[(f["persona"], f["dia"])][f["comida"]] = int(f["clase"] == "OK")
X = np.array([[b["desayuno"], b["almuerzo"], b["cena"]] for b in blo.values() if len(b) == 3])
if len(X):
    k = 3; Cj, Ri, N = X.sum(0), X.sum(1), X.sum()
    Q = (k - 1) * (k * (Cj ** 2).sum() - N ** 2) / (k * N - (Ri ** 2).sum())
    OUT["cochran_turnos"] = {"bloques": int(len(X)), "exito_1er_intento_%": dict(zip(("desayuno", "almuerzo", "cena"), (100 * X.mean(0)).round(1).tolist())),
                             "Q": round(float(Q), 2), "gl": 2, "p": f"{1 - st.chi2.cdf(Q, 2):.2e}"}

# FNIR por sexo (atributo "Male" de LFW) con prueba exacta de Fisher
import os
_base = os.path.dirname(os.path.abspath(sys.argv[1]))
attr = next((c for c in (os.path.join(_base, "lfw_attributes.txt"), os.path.join(_base, "datos", "lfw_attributes.txt"),
                         os.path.join("datos_pruebas", "lfw_attributes.txt")) if os.path.exists(c)), "")
if os.path.exists(attr):
    L = open(attr).read().splitlines(); H = L[1].lstrip("#").strip().split("	"); male = collections.defaultdict(list)
    for l in L[2:]:
        r = dict(zip(H, l.split("	"))); male[r["person"].replace(" ", "_")].append(float(r["Male"]))
    g = collections.defaultdict(lambda: [0, 0])
    for f in AUD:
        if f["tipo"] == "registrado" and f["persona"] in male:
            sx = "H" if np.mean(male[f["persona"]]) > 0 else "M"; g[sx][1] += 1; g[sx][0] += f["clase"] != "OK"
    if "H" in g and "M" in g:
        o, pf = st.fisher_exact([[g["H"][0], g["H"][1] - g["H"][0]], [g["M"][0], g["M"][1] - g["M"][0]]])
        OUT["FNIR_por_sexo"] = {"H": {"FNIR_%": pct(*g["H"]), "IC95": wilson(*g["H"]), "n": g["H"][1]},
                                "M": {"FNIR_%": pct(*g["M"]), "IC95": wilson(*g["M"]), "n": g["M"][1]},
                                "odds_ratio": round(float(o), 2), "p_Fisher": f"{pf:.3g}"}
OUT["muestra"] = {"registrados": len(RES["galeria_ids"]), "impostores": RES["impostores"]}
json.dump(OUT, open(sys.argv[1].replace(".json", "_metricas.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(OUT, indent=1, ensure_ascii=False, default=str))
