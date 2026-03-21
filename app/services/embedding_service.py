from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.embedding import Embedding
from app.models.comensal import Comensal
from app.core.config import settings
import numpy as np

class EmbeddingService:
    
    async def save_embedding(
        self, 
        db: AsyncSession, 
        comensal_id: int, 
        embedding: np.ndarray,
        calidad: float,
        fuente: str,
        es_principal: bool = False
    ):
        """Guarda un embedding en la BD"""
        emb = Embedding(
            comensal_id=comensal_id,
            embedding=embedding.tolist(),
            calidad=calidad,
            fuente=fuente,
            es_principal=es_principal
        )
        db.add(emb)
        await db.commit()
        return emb
    
    async def find_match(
        self, 
        db: AsyncSession, 
        embedding: np.ndarray
    ):
        """Busca el embedding más cercano usando pgvector"""
        # Búsqueda vectorial con distancia coseno
        query = select(
            Embedding.comensal_id,
            Embedding.id,
            Embedding.embedding.cosine_distance(embedding.tolist()).label("distance")
        ).order_by("distance").limit(1)
        
        result = await db.execute(query)
        match = result.first()
        
        if match and match.distance < settings.FACE_THRESHOLD:
            # Obtener datos del comensal
            comensal = await db.get(Comensal, match.comensal_id)
            return {
                "comensal": comensal,
                "distance": match.distance,
                "embedding_id": match.id
            }
        
        return None
    
    async def count_embeddings(self, db: AsyncSession, comensal_id: int) -> int:
        """Cuenta embeddings de un comensal"""
        result = await db.execute(
            select(func.count(Embedding.id)).where(Embedding.comensal_id == comensal_id)
        )
        return result.scalar()

embedding_service = EmbeddingService()