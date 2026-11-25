"""
Metricas ISO/IEC 19795-1 y 30107-3 a partir de la Prueba 1 (eval_100.json/.npz), con los resumenes
de las pruebas A y B (lat_resultados.json). No corre pruebas nuevas.
Uso: python scripts/pruebas/metricas_iso.py <carpeta con eval_100.json/.npz, lat_resultados.json y datos/>
"""
import sys, os, json, math, csv, collections
import numpy as np
from scipy import stats as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

S = sys.argv[1]
R = json.load(open(os.path.join(S, "eval_100.json")))
Z = np.load(os.path.join(S, "eval_100.npz")); GAL, DIST = Z["gal_mat"], Z["dist_embs"]
P = R["gal_pers"]; THR = R["config"]["MATCH_THRESHOLD"]
LAT = json.load(open(os.path.join(S, "lat_resultados.json"), encoding="utf-8"))
OUT = {}

def wilson(k, n, z=1.96):
    if not n: return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * (c - m) / d, 2), round(100 * (c + m) / d, 2)]
def pc(k, n): return round(100 * k / n, 2) if n else None
def cp_sup(k, n): return round(100 * st.beta.ppf(0.975, k + 1, n - k), 2) if k < n else 100.0
def tasa(k, n): return {"valor_%": pc(k, n), "IC95_%": wilson(k, n), "k": k, "n": n}

# ---------------------------------------------------------------- 1. Registro y captura
reg = R["registro"]
n_reg, ok_reg = len(reg), sum(r["ok"] for r in reg)
OUT["1_registro"] = {
    "FTE": tasa(n_reg - ok_reg, n_reg),
    "FTE_por_fuente": {f: tasa(sum(not r["ok"] for r in reg if r["fuente"] == f), sum(r["fuente"] == f for r in reg)) for f in ("crema", "lfw")},
    "calidad_media_registro": {
        "fotogramas_validos_de_15_media": round(np.mean([r["validos"] for r in reg if r["ok"]]), 2),
        "fotogramas_validos_de_15_DE": round(np.std([r["validos"] for r in reg if r["ok"]], ddof=1), 2),
        "nota": "Puntaje de calidad = fotogramas que pasan deteccion>=0.80, tamano>=80px, luz y liveness."},
}
Q = R["consultas"]
for q in Q:      # quien no logro registrarse cuenta como desconocido
    if q["verdadero"] and q["verdadero"] not in P: q["verdadero"] = None
leg = [q for q in Q if q["etiqueta"] == "real" and q["verdadero"]]
imp = [q for q in Q if q["etiqueta"] == "real" and not q["verdadero"]]
todos = leg + imp
OUT["1_registro"]["FTA"] = tasa(sum(q["estado"] != "OK" for q in todos), len(todos))
OUT["1_registro"]["FTA_motivos"] = dict(collections.Counter(q["estado"] for q in todos if q["estado"] != "OK"))

# ---------------------------------------------------------------- 2-3. Identificacion (umbral actual)
def decidir(q, thr=THR):
    if q["estado"] != "OK": return None
    E = np.asarray(q["embs"], np.float32); D = 1 - E @ GAL.T
    j = D.argmin(1); d = D[np.arange(len(j)), j]
    v = collections.Counter(P[k] for k, dd in zip(j, d) if dd < thr)
    if not v or v.most_common(1)[0][1] < math.ceil(q["validos"] * 0.6): return None
    return v.most_common(1)[0][0]

def bloque_identificacion(L, I, thr=THR):
    OK = NR = CF = 0
    for q in L:
        w = decidir(q, thr)
        if w is None: NR += 1
        elif w == q["verdadero"]: OK += 1
        else: CF += 1
    FA = sum(decidir(q, thr) is not None for q in I)
    N = len(L); prec = OK / (OK + CF + FA) if OK + CF + FA else 0; rec = OK / N if N else 0
    pers = collections.defaultdict(list)
    for q in I: pers[q["persona"]].append(decidir(q, thr) is not None)
    return {"N_leg": N, "OK": OK, "NR": NR, "CF": CF, "N_imp": len(I), "FA": FA,
            "tasa_identificacion_correcta": tasa(OK, N), "FNIR": tasa(NR + CF, N), "FRR": tasa(NR, N),
            "tasa_confusion": {**tasa(CF, N), "limite_sup_95_%": cp_sup(CF, N)},
            "recall_%": round(100 * rec, 2), "precision_%": round(100 * prec, 2),
            "F1_%": round(200 * prec * rec / (prec + rec), 2) if prec + rec else 0,
            "FPIR_por_intento": {**tasa(FA, len(I)), "limite_sup_95_%": cp_sup(FA, len(I))},
            "FPIR_por_persona": {**tasa(sum(any(v) for v in pers.values()), len(pers)),
                                 "limite_sup_95_%": cp_sup(sum(any(v) for v in pers.values()), len(pers))}}

OUT["2_3_identificacion"] = {
    "global": bloque_identificacion(leg, imp),
    "kiosco_rafaga_CREMA": bloque_identificacion([q for q in leg if q["fuente"] == "crema"], [q for q in imp if q["fuente"] == "crema"]),
    "fotos_otros_anios_LFW": bloque_identificacion([q for q in leg if q["fuente"] == "lfw"], [q for q in imp if q["fuente"] == "lfw"]),
}
# Tasa correcta total contando reintentos por deteccion de vida (capas activas, medido en estas rafagas reales)
kio = [q for q in leg if q["fuente"] == "crema" and q.get("liv")]
rech_vida = sum(not (q["liv"]["minifasnet"] >= 0.5 and q["liv"]["prof_estado"] != "plano") for q in kio)
ok_kio = OUT["2_3_identificacion"]["kiosco_rafaga_CREMA"]["OK"]
OUT["2_3_identificacion"]["tasa_correcta_total_con_reintentos_vida"] = {
    "valor_%": pc(ok_kio, len([q for q in leg if q["fuente"] == "crema"]) + rech_vida),
    "reintentos_por_vida": rech_vida, "nota": "Kiosco CREMA-D con capas de vida activas (MiniFASNet>=0.5 y profundidad)."}

# ---------------------------------------------------------------- 4. Umbral y distancias (nivel par: fotograma vs plantilla)
gen_d, imp_d = [], []
for q in Q:
    if q["etiqueta"] != "real" or q["estado"] != "OK": continue
    E = np.asarray(q["embs"], np.float32); D = 1 - E @ GAL.T
    if q["verdadero"]:
        k = P.index(q["verdadero"]); gen_d += D[:, k].tolist(); imp_d += np.delete(D, k, 1).ravel().tolist()
    else:
        imp_d += D.ravel().tolist()
imp_d += (1 - DIST @ GAL.T).ravel().tolist()          # 1,834 distractores contra los 94 registrados
gen_d, imp_d = np.array(gen_d), np.array(imp_d)
def fmr(t): return float((imp_d < t).mean())
def fnmr(t): return float((gen_d >= t).mean())
ts = np.linspace(0.05, 1.2, 2301)
FMR = np.array([fmr(t) for t in ts]); FNMR = np.array([fnmr(t) for t in ts])
i_eer = int(np.argmin(np.abs(FMR - FNMR)))
OUT["4_umbral_distancias"] = {
    "pares_misma_persona": int(len(gen_d)), "pares_personas_distintas": int(len(imp_d)),
    "distancia_misma_persona": {"media": round(gen_d.mean(), 3), "DE": round(gen_d.std(ddof=1), 3), "p95": round(np.percentile(gen_d, 95), 3), "max": round(gen_d.max(), 3)},
    "distancia_personas_distintas": {"media": round(imp_d.mean(), 3), "DE": round(imp_d.std(ddof=1), 3), "p0.01": round(np.percentile(imp_d, 0.01), 3), "min": round(imp_d.min(), 3)},
    "en_umbral": {str(t): {"FMR": f"{fmr(t):.2e}", "FMR_limite_sup_95": f"{st.beta.ppf(0.975, int((imp_d < t).sum()) + 1, len(imp_d) - int((imp_d < t).sum())):.2e}",
                           "FNMR_%": round(100 * fnmr(t), 2), "confianza_equivalente_%": round((1 - t / 2) * 100, 1)}
                  for t in (0.35, 0.40, 0.45, 0.50, 0.55)},
    "EER_%": round(100 * (FMR[i_eer] + FNMR[i_eer]) / 2, 3), "EER_umbral": round(float(ts[i_eer]), 3),
    "separacion_d_prime": round(abs(imp_d.mean() - gen_d.mean()) / math.sqrt((imp_d.var() + gen_d.var()) / 2), 2),
    # estimacion de la cola (FMR muy pequenos que no se observan con este tamano de muestra)
    "FMR_estimado_cola_normal_0.45": f"{st.norm.cdf((0.45 - imp_d.mean()) / imp_d.std()):.2e}",
}
# Figura DET + histogramas
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].hist(gen_d, bins=60, alpha=.7, density=True, label=f"Misma persona (n={len(gen_d):,})", color="#2a9d8f")
ax[0].hist(imp_d, bins=120, alpha=.6, density=True, label=f"Personas distintas (n={len(imp_d):,})", color="#e76f51")
ax[0].axvline(THR, color="k", ls="--", label=f"Umbral {THR}"); ax[0].set_xlabel("Distancia coseno"); ax[0].set_ylabel("Densidad")
ax[0].set_title("Distribucion de distancias"); ax[0].legend(fontsize=8)
m = (FMR > 0) & (FNMR > 0)
ax[1].loglog(FMR[m] * 100, FNMR[m] * 100, color="#264653"); ax[1].set_xlabel("FMR (%)"); ax[1].set_ylabel("FNMR (%)")
ax[1].set_title(f"Curva DET (EER = {OUT['4_umbral_distancias']['EER_%']}%)"); ax[1].grid(True, which="both", alpha=.3)
for t in (0.40, 0.45, 0.50):
    a, b = fmr(t), fnmr(t)
    if a > 0 and b > 0: ax[1].plot(a * 100, b * 100, "o"); ax[1].annotate(f"{t}", (a * 100, b * 100))
plt.tight_layout(); plt.savefig(os.path.join(S, "curva_det.png"), dpi=130)
OUT["4_umbral_distancias"]["nota_DET"] = "Con este tamano de muestra no hay pares de personas distintas bajo ~0.69: la curva DET solo existe a la derecha del umbral; en 0.45 el FMR observado es 0 y se reporta su limite superior."

# ---------------------------------------------------------------- 5. Deteccion de vida
crema_real = [q for q in Q if q.get("liv") and q["etiqueta"] == "real" and q["fuente"] == "crema"]
foto = [q for q in Q if q.get("liv") and q["etiqueta"] == "foto_mano"]
cel = R["celeba"]; cel_ok = [c for c in cel if c["score"] is not None and c.get("detalle") == "ok"]
ata_cel = [c for c in cel_ok if c["clase"] == "spoof"]; viv_cel = [c for c in cel_ok if c["clase"] == "live"]
def pasa_capas(q): return q["liv"]["minifasnet"] >= 0.5 and q["liv"]["prof_estado"] != "plano"
OUT["5_vida"] = {
    "APCER_por_tipo": {
        "foto_en_la_mano_modo_actual": tasa(sum(q["liv"]["es_real"] for q in foto), len(foto)),
        "foto_en_la_mano_capas_activas": {**tasa(sum(pasa_capas(q) for q in foto), len(foto)), "limite_sup_95_%": cp_sup(sum(pasa_capas(q) for q in foto), len(foto))},
        "impresion_pantalla_CelebA_Spoof_solo_MiniFASNet": tasa(sum(c["score"] >= 0.5 for c in ata_cel), len(ata_cel)),
        "video_en_pantalla_sesion_anterior": {"valor_%": 82.4, "nota": "profundidad sola; el fondo de referencia lo baja a 0% (ver CLAUDE.md)"},
    },
    "APCER_global_modo_actual": tasa(sum(q["liv"]["es_real"] for q in foto) + sum(c["score"] >= 0.5 for c in ata_cel), len(foto) + len(ata_cel)),
    "APCER_global_capas_activas": {"valor_%": None, "nota": "Las imagenes de CelebA-Spoof son fotos sueltas: no se les puede medir la profundidad. Peor tipo con capas: ver por tipo."},
    "BPCER_kiosco_real_CREMA": {"MiniFASNet": tasa(sum(q["liv"]["minifasnet"] < 0.5 for q in crema_real), len(crema_real)),
                                "capas_activas": tasa(sum(not pasa_capas(q) for q in crema_real), len(crema_real)),
                                "modo_actual": tasa(sum(not q["liv"]["es_real"] for q in crema_real), len(crema_real))},
    "BPCER_CelebA_Spoof_vivos_MiniFASNet": tasa(sum(c["score"] < 0.5 for c in viv_cel), len(viv_cel)),
}

# ---------------------------------------------------------------- 6. Tiempos
t_an = np.array([q["seg"] for q in Q if q["tipo"] == "rafaga"])
t_e2e = np.array([e["seg"] for e in R["e2e"]])
t_reg = np.array([r["seg"] for r in reg])
def tiempos(t):
    return {"n": int(len(t)), "mediana_s": round(float(np.median(t)), 3), "RIC_s": [round(float(np.percentile(t, 25)), 3), round(float(np.percentile(t, 75)), 3)],
            "media_s": round(float(t.mean()), 3), "DE_s": round(float(t.std(ddof=1)), 3), "p90_s": round(float(np.percentile(t, 90)), 3)}
OUT["6_tiempos"] = {
    "analisis_rafaga_5_fotos": tiempos(t_an),
    "endpoint_completo_HTTP_BD": tiempos(t_e2e),
    "registro_15_fotos": tiempos(t_reg),
    "latencia_por_etapa_ms_por_foto": {"deteccion+106_puntos+ArcFace": "145-177 (medido en la sesion de optimizacion)",
                                        "busqueda_en_cache": "<1 (producto matricial)", "nota": "No se cronometro cada etapa por separado en esta prueba."},
    "personas_por_minuto_solo_sistema": round(60 / float(t_e2e.mean() + 0.9), 1),
    "nota_personas_por_minuto": "Incluye ~0.9 s de captura de la rafaga (5 fotos x 180 ms). No incluye el tiempo de caminar hasta el kiosco.",
    "tiempo_manual": "NO MEDIDO: hay que cronometrarlo en la fila del comedor (sin esto no hay reduccion de tiempo ni delta de Glass).",
}

# ---------------------------------------------------------------- 7. Equidad y condiciones
demo = {int(r["ActorID"]): r for r in csv.DictReader(open(os.path.join(S, "datos", "cremad_demografia.csv")))}
Lh = open(os.path.join(S, "datos", "lfw_attributes.txt")).read().splitlines(); Hh = Lh[1].lstrip("#").strip().split("\t")
male = collections.defaultdict(list)
for l in Lh[2:]:
    r = dict(zip(Hh, l.split("\t"))); male[r["person"].replace(" ", "_")].append(float(r["Male"]))
_R = os.path.join(S, "datos", "lfw", "lfw_home", "lfw_funneled"); nombres_lfw = sorted(d for d in os.listdir(_R) if os.path.isdir(os.path.join(_R, d)))
def sexo(q):
    p = q["persona"]
    if p.startswith("C"): return demo[int(p[1:])]["Sex"] == "Male" and "H" or "M"
    nm = nombres_lfw[int(p[1:])] if p[1:].isdigit() and int(p[1:]) < len(nombres_lfw) else None
    return ("H" if np.mean(male[nm]) > 0 else "M") if nm in male else None
def etnia(q):
    p = q["persona"]
    return ("Hispano" if demo[int(p[1:])]["Ethnicity"] == "Hispanic" else "No hispano") if p.startswith("C") else None

def fnir_grupos(L, fn):
    g = collections.defaultdict(lambda: [0, 0])
    for q in L:
        k = fn(q)
        if k is None: continue
        g[k][1] += 1; g[k][0] += decidir(q) != q["verdadero"]
    return g
def comparar(g, a, b):
    if a not in g or b not in g: return None
    tab = [[g[a][0], g[a][1] - g[a][0]], [g[b][0], g[b][1] - g[b][0]]]
    orr, p = st.fisher_exact(tab)
    return {a: tasa(*g[a]), b: tasa(*g[b]), "odds_ratio": round(float(orr), 2) if np.isfinite(orr) else "inf", "p_Fisher": round(float(p), 4)}
OUT["7_equidad"] = {
    "FNIR_por_sexo_global": comparar(fnir_grupos(leg, sexo), "H", "M"),
    "FNIR_por_sexo_kiosco_CREMA": comparar(fnir_grupos([q for q in leg if q["fuente"] == "crema"], sexo), "H", "M"),
    "FNIR_por_etnia_kiosco_CREMA": comparar(fnir_grupos([q for q in leg if q["fuente"] == "crema"], etnia), "Hispano", "No hispano"),
    "FNIR_por_turno": "Pendiente: sale de la simulacion del comedor (10 dias x desayuno/almuerzo/cena), que esta corriendo.",
    "FNIR_por_condicion_lentes_prueba_A": {k: round(100 - v["reconoce_%"], 2) for k, v in LAT["A"]["resultados"].items()},
}
json.dump(OUT, open(os.path.join(S, "metricas_iso.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(OUT, indent=1, ensure_ascii=False, default=str))
