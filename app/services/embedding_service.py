from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.embedding import Embedding
from app.models.comensal import Comensal
import numpy as np


class EmbeddingService:

    async def save_embedding(
        self,
        db: AsyncSession,
        comensal_id: int,
        embedding: np.ndarray,
        calidad: float,
        fuente: str,
        es_principal: bool = False,
        es_promedio: bool = False,
    ):
        """Guarda un embedding individual en la BD."""
        emb = Embedding(
            comensal_id=comensal_id,
            embedding=embedding.tolist(),
            calidad=calidad,
            fuente=fuente,
            es_principal=es_principal,
            es_promedio=es_promedio,
        )
        db.add(emb)
        return emb

    async def save_batch(
        self,
        db: AsyncSession,
        comensal_id: int,
        embeddings: list,
        avg_embedding: np.ndarray,
        fuente: str,
    ):
        """
        Guarda el embedding promedio (es_promedio=True) y los individuales.
        El promedio se usa para matching; los individuales se conservan
        para futura mejora del modelo.
        """
        # Embedding promedio (principal)
        avg_emb = Embedding(
            comensal_id=comensal_id,
            embedding=avg_embedding.tolist(),
            calidad=1.0,
            fuente=fuente,
            es_principal=True,
            es_promedio=True,
        )
        db.add(avg_emb)

        # Embeddings individuales
        for emb_array in embeddings:
            individual = Embedding(
                comensal_id=comensal_id,
                embedding=emb_array.tolist(),
                calidad=0.8,
                fuente=fuente,
                es_principal=False,
                es_promedio=False,
            )
            db.add(individual)

    async def update_average_embedding(
        self,
        db: AsyncSession,
        comensal_id: int,
        new_avg_embedding: np.ndarray,
    ):
        """Actualiza el embedding promedio en la BD (para auto-enriquecimiento)."""
        result = await db.execute(
            select(Embedding).where(
                Embedding.comensal_id == comensal_id,
                Embedding.es_promedio == True,
            )
        )
        emb_record = result.scalar_one_or_none()

        if emb_record:
            emb_record.embedding = new_avg_embedding.tolist()
            await db.commit()

    async def count_embeddings(self, db: AsyncSession, comensal_id: int) -> int:
        """Cuenta embeddings de un comensal."""
        result = await db.execute(
            select(func.count(Embedding.id)).where(
                Embedding.comensal_id == comensal_id
            )
        )
        return result.scalar()


embedding_service = EmbeddingService()
