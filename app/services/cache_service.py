"""
Cache en memoria de embeddings para comparacion vectorial rapida.

Carga todos los embeddings promedio desde PostgreSQL/pgvector al iniciar
y los mantiene en una matriz numpy para comparacion en O(1) via dot product.
"""

import numpy as np
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.embedding import Embedding
from app.models.comensal import Comensal
from loguru import logger


class EmbeddingCache:
    _instance = None

    def __init__(self):
        self._data = {}       # comensal_id -> {nombre, apellido, dni, embedding}
        self._matrix = None   # (N, 512) numpy matrix para comparacion vectorizada
        self._ids = []        # lista ordenada de comensal_ids (misma fila que _matrix)
        self._loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def load(self, db: AsyncSession):
        """Carga todos los embeddings promedio activos desde la BD."""
        logger.info("Cargando embeddings en cache...")

        result = await db.execute(
            select(Embedding, Comensal)
            .join(Comensal, Embedding.comensal_id == Comensal.id)
            .where(Embedding.es_promedio == True)
            .where(Comensal.estado == True)
        )

        self._data = {}
        embeddings_list = []
        ids_list = []

        for emb, comensal in result:
            embedding_array = np.array(emb.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm

            self._data[comensal.id] = {
                "nombre": comensal.nombre,
                "apellido": comensal.apellido,
                "dni": comensal.dni,
                "embedding": embedding_array,
            }
            embeddings_list.append(embedding_array)
            ids_list.append(comensal.id)

        if embeddings_list:
            self._matrix = np.vstack(embeddings_list)
        else:
            self._matrix = None

        self._ids = ids_list
        self._loaded = True
        logger.info(f"Cache cargado: {len(self._ids)} comensales")

    def find_match(self, embedding: np.ndarray, threshold: float) -> Optional[dict]:
        """
        Busca el embedding mas cercano usando distancia coseno vectorizada.
        Como los embeddings estan L2-normalizados, cosine_similarity = dot product.
        """
        if not self._loaded or self._matrix is None or len(self._ids) == 0:
            return None

        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        # Una sola operacion matricial para comparar contra todos los usuarios
        similarities = np.dot(self._matrix, emb)
        distances = 1.0 - similarities

        min_idx = int(np.argmin(distances))
        min_distance = float(distances[min_idx])

        if min_distance < threshold:
            comensal_id = self._ids[min_idx]
            info = self._data[comensal_id]
            return {
                "comensal_id": comensal_id,
                "distance": min_distance,
                "nombre": info["nombre"],
                "apellido": info["apellido"],
                "dni": info["dni"],
            }
        return None

    def check_duplicate(self, embedding: np.ndarray, threshold: float) -> Optional[dict]:
        """Verifica si un rostro ya existe en el cache (anti-duplicado)."""
        return self.find_match(embedding, threshold)

    def add(self, comensal_id: int, info: dict, embedding: np.ndarray):
        """Agrega un nuevo usuario al cache (llamado despues de registro exitoso)."""
        emb = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        self._data[comensal_id] = {
            "nombre": info["nombre"],
            "apellido": info["apellido"],
            "dni": info["dni"],
            "embedding": emb,
        }

        self._ids.append(comensal_id)

        if self._matrix is not None:
            self._matrix = np.vstack([self._matrix, emb.reshape(1, -1)])
        else:
            self._matrix = emb.reshape(1, -1)

        logger.info(f"Cache actualizado: +comensal {comensal_id} (total: {len(self._ids)})")

    def enrich_embedding(self, comensal_id: int, new_embedding: np.ndarray, weight: float = 0.15):
        """
        Enriquece el embedding promedio de un usuario mezclando con un nuevo embedding.
        Esto permite que el sistema 'aprenda' cambios de apariencia (lentes, barba, gorra, etc.)

        Formula: new_avg = normalize((1-weight) * old_avg + weight * new_emb)
        weight=0.15 significa que el nuevo embedding influye un 15% en el promedio.
        """
        if comensal_id not in self._data:
            return False

        new_emb = new_embedding.astype(np.float32)
        norm = np.linalg.norm(new_emb)
        if norm > 0:
            new_emb = new_emb / norm

        old_emb = self._data[comensal_id]["embedding"]

        # Mezcla ponderada
        blended = (1.0 - weight) * old_emb + weight * new_emb
        blended_norm = np.linalg.norm(blended)
        if blended_norm > 0:
            blended = blended / blended_norm

        # Actualizar en cache
        self._data[comensal_id]["embedding"] = blended

        # Actualizar en la matriz
        if comensal_id in self._ids:
            idx = self._ids.index(comensal_id)
            self._matrix[idx] = blended

        logger.info(f"Embedding enriquecido: comensal {comensal_id} (weight={weight})")
        return True

    @property
    def count(self):
        return len(self._ids)
