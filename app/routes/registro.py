from fastapi import APIRouter, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.comensal import Comensal
from app.services.face_service import face_service
from app.services.embedding_service import embedding_service
import cv2
import base64
import numpy as np
from sqlalchemy import select

router = APIRouter(prefix="/api", tags=["registro"])

@router.post("/registrar_webcam")
async def registrar_webcam(
    dni: str = Form(...),
    nombre: str = Form(...),
    apellido: str = Form(...),
    email: str = Form(None),
    foto_base64: str = Form(...),
    metodo: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Registra comensal desde webcam O foto subida"""
    
    try:
        # Decodificar imagen base64
        img_data = base64.b64decode(foto_base64.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detectar rostro
        result = face_service.detect_face(image)
        
        if result is None:
            return JSONResponse(
                {"success": False, "message": "No se detectó rostro en la imagen"}, 
                status_code=400
            )
        
        if result["is_spoofing"]:
            return JSONResponse({
                "success": False, 
                "message": "⚠️ ALERTA: Intento de spoofing detectado"
            }, status_code=400)
        
        # Verificar DNI único
        existing = await db.execute(select(Comensal).where(Comensal.dni == dni))
        if existing.scalar_one_or_none():
            return JSONResponse(
                {"success": False, "message": "DNI ya registrado"}, 
                status_code=400
            )
        
        # Crear comensal CON FOTO GUARDADA
        comensal = Comensal(
            dni=dni, 
            nombre=nombre, 
            apellido=apellido,
            email=email,
            foto_registro=foto_base64  # ← GUARDAR FOTO
        )
        db.add(comensal)
        await db.flush()
        
        # Guardar embedding
        calidad = face_service.calculate_quality(result["face"])
        await embedding_service.save_embedding(
            db, comensal.id, result["embedding"], 
            calidad, f"registro_{metodo}", es_principal=True
        )
        
        await db.commit()
        
        return {
            "success": True,
            "message": f"✅ {nombre} {apellido} registrado correctamente",
            "liveness_score": round(result["liveness_score"], 2),
            "calidad": round(calidad, 2),
            "metodo": metodo
        }
        
    except Exception as e:
        return JSONResponse(
            {"success": False, "message": f"Error: {str(e)}"}, 
            status_code=500
        )