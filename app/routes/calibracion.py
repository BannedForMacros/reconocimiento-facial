"""
Calibracion del liveness en el kiosco real.

Guarda rafagas etiquetadas (real / foto / celular / video) para medir cada capa
con la camara, la pantalla y la luz reales del comedor, y comparar los modos
"profundidad" y "luz". Solo se permite desde la propia maquina (localhost),
porque almacena imagenes de rostros.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from app.routes.rafaga import analizar_rafaga, colores_de_desafio, decodificar
from app.services.multiframe_liveness import referencia_fondo

router = APIRouter(tags=["calibracion"])
templates = Jinja2Templates(directory="templates")

DIR = Path("data") / "calibracion"
ETIQUETAS = {"real", "foto", "celular", "video"}
LOCAL = {"127.0.0.1", "::1", "localhost"}


def _solo_local(request: Request):
    if not request.client or request.client.host not in LOCAL:
        raise HTTPException(status_code=403, detail="Calibracion solo disponible en el equipo local")


@router.get("/calibracion", response_class=HTMLResponse)
async def calibracion_view(request: Request):
    _solo_local(request)
    return templates.TemplateResponse("calibracion.html", {"request": request})


@router.post("/api/calibracion/muestra")
async def guardar_muestra(
    request: Request,
    etiqueta: str = Form(...),
    fotos: List[UploadFile] = File(...),
    desafio_id: Optional[str] = Form(None),
    indices: Optional[str] = Form(None),
):
    _solo_local(request)
    if etiqueta not in ETIQUETAS:
        raise HTTPException(status_code=400, detail="Etiqueta invalida")

    blobs = [await f.read() for f in fotos]
    images = decodificar(blobs)
    colores, err = colores_de_desafio(desafio_id, indices, len(images))
    if err:
        raise HTTPException(status_code=400, detail=err)

    kiosco = request.client.host if request.client else "default"
    r = await run_in_threadpool(analizar_rafaga, images, colores, referencia_fondo.obtener(kiosco))
    modo = "luz" if colores else "profundidad"

    carpeta = DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{modo}_{etiqueta}"
    carpeta.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(blobs):
        (carpeta / f"{i:02d}.jpg").write_bytes(b)
    meta = {"etiqueta": etiqueta, "modo": modo, "colores": colores,
            "estado": r["estado"], "senales": r.get("senales")}
    (carpeta / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"ok": True, "modo": modo, "estado": r["estado"], "senales": r.get("senales")}


@router.get("/api/calibracion/resumen")
async def resumen(request: Request):
    """Conteo rapido de muestras guardadas por modo y etiqueta."""
    _solo_local(request)
    conteo = {}
    for meta in DIR.glob("*/meta.json"):
        m = json.loads(meta.read_text(encoding="utf-8"))
        k = f"{m['modo']}/{m['etiqueta']}"
        conteo[k] = conteo.get(k, 0) + 1
    return conteo
