"""Confianza de deteccion (det_score) de las rafagas que fallaron por calidad vs las buenas.
Uso: python scripts/pruebas/confianza_deteccion.py <carpeta con eval_100.json/.npz y datos/>"""
import sys, os, json, numpy as np, cv2, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,".")
from loguru import logger; logger.remove()
from app.services.face_service import face_service
S=sys.argv[1]; R=json.load(open(S+"/eval_100.json")); Z=np.load(S+"/eval_100.npz"); G=Z["gal_mat"]; P=R["gal_pers"]
def frames(p):
    cap=cv2.VideoCapture(p); out=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        out.append(cv2.resize(f,(640,480)))
    return out
malos=[(q["persona"],q["clip"]) for q in R["consultas"] if q["fuente"]=="crema" and q["estado"]!="OK" and q["clip"]!="foto"]
buenos=[(q["persona"],q["clip"]) for q in R["consultas"] if q["fuente"]=="crema" and q["estado"]=="OK" and q["clip"]=="DFA_NEU_XX"][:25]
for nombre,lst in (("fallidos",malos),("buenos",buenos)):
    sc=[]; dist=[]
    for pers,clip in lst:
        fr=frames(os.path.join(S,"datos","cremad2",f"{pers[1:]}_{clip}.flv"))
        for i in range(5):
            d=face_service.detect_face(fr[min(20+5*i,len(fr)-1)])
            if d is None: continue
            sc.append(d["det_score"])
            if pers in P: dist.append(float(1-d["embedding"]@G[P.index(pers)]))
    sc=np.array(sc); print(nombre,"n",len(sc),"det_score media",sc.mean().round(3),"min",sc.min().round(3),
      ">=0.80",(sc>=.8).mean().round(2),">=0.70",(sc>=.7).mean().round(2),">=0.60",(sc>=.6).mean().round(2),
      "| dist a su registro: media",np.mean(dist).round(3) if dist else None,"max",np.max(dist).round(3) if dist else None)
