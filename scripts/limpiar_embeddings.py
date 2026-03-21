"""
Script de limpieza para re-registro de comensales.

Ejecutar ANTES de usar el nuevo sistema de registro multi-frame.
Este script:
1. Agrega la columna es_promedio a la tabla embeddings (si no existe)
2. Elimina TODOS los embeddings existentes
3. Resetea las estadisticas de marcaciones de los comensales
4. NO elimina los comensales ni sus datos personales

USO:
    python scripts/limpiar_embeddings.py
"""

import psycopg2
import sys

# Conexion sincrona (psycopg2) - ajustar si es necesario
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "comedor_db"
DB_USER = "postgres"
DB_PASS = "postgres"


def limpiar():
    print("=" * 50)
    print("  LIMPIEZA DE EMBEDDINGS - Sistema Comedor")
    print("=" * 50)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()

    # 1. Agregar columna es_promedio si no existe
    print("\n[1/4] Verificando schema de BD...")
    cur.execute("""
        ALTER TABLE embeddings
        ADD COLUMN IF NOT EXISTS es_promedio BOOLEAN DEFAULT FALSE;
    """)
    print("  -> Columna es_promedio: OK")

    # 2. Contar embeddings actuales
    cur.execute("SELECT COUNT(*) FROM embeddings;")
    count = cur.fetchone()[0]
    print(f"\n[2/4] Embeddings actuales: {count}")

    # 3. Eliminar todos los embeddings
    cur.execute("DELETE FROM embeddings;")
    print(f"  -> Eliminados: {count} embeddings")

    # 4. Resetear estadisticas de comensales
    cur.execute("""
        UPDATE comensales
        SET total_marcaciones = 0,
            ultima_marcacion = NULL;
    """)
    cur.execute("SELECT COUNT(*) FROM comensales;")
    comensales = cur.fetchone()[0]
    print(f"\n[3/4] Estadisticas reseteadas para {comensales} comensales")

    # 5. Limpiar intentos de acceso (opcional)
    cur.execute("DELETE FROM intentos_acceso;")
    print("[4/4] Intentos de acceso limpiados")

    conn.commit()
    cur.close()
    conn.close()

    print("\n" + "=" * 50)
    print("  LIMPIEZA COMPLETADA")
    print("=" * 50)
    print()
    print("Siguiente paso:")
    print("  1. Inicia el servidor: python -m app.main")
    print("  2. Cada comensal debe re-registrarse en /registro")
    print("     con el nuevo flujo de 15 capturas faciales.")
    print()


if __name__ == "__main__":
    print()
    resp = input("Esto eliminara TODOS los embeddings. Los comensales deberan re-registrarse.\n\nContinuar? (si/no): ")
    if resp.strip().lower() in ("si", "s", "yes", "y"):
        limpiar()
    else:
        print("Cancelado.")
        sys.exit(0)
