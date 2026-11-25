"""
Descarga los datasets publicos que usan las pruebas en datos_pruebas/ (ignorada por git, ~1.5 GB).

Uso (desde la raiz del proyecto):  python scripts/pruebas/descargar_datos.py

| Dataset | Para que | Licencia / uso |
|---|---|---|
| LFW funneled (13,233 fotos, 5,749 personas) | registrados e impostores | investigacion |
| Atributos de LFW (Columbia) | lentes / lentes oscuros / sexo | investigacion |
| FairFace val + train/0 (padding 1.25) | impostores latinos y selector de poblacion | CC BY 4.0 |
| CelebA-Spoof (2 partes del test) | ataques reales (impresion, pantalla) | investigacion |
| CREMA-D demografia | sexo y etnia de los actores (prueba 1) | ODbL |

No se suben al repositorio: son fotos de rostros de terceros.
"""
import os, sys, urllib.request

DEST = sys.argv[1] if len(sys.argv) > 1 else "datos_pruebas"
HF = "https://huggingface.co"
ARCHIVOS = {
    "lfw_attributes.txt": "https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt",
    "fairface/val.parquet": f"{HF}/api/datasets/HuggingFaceM4/FairFace/parquet/1.25/validation/0.parquet",
    "fairface/train0.parquet": f"{HF}/api/datasets/HuggingFaceM4/FairFace/parquet/1.25/train/0.parquet",
    "celebaspoof/test-00005-of-00065.parquet": f"{HF}/datasets/Ar4ikov/celebA_spoof/resolve/main/data/test-00005-of-00065-4a3790a330e0cf11.parquet",
    "celebaspoof/test-00045-of-00065.parquet": f"{HF}/datasets/Ar4ikov/celebA_spoof/resolve/main/data/test-00045-of-00065-748530e1c46bbc70.parquet",
    "cremad_demografia.csv": "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/VideoDemographics.csv",
}


def bajar(url, destino):
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        print("ya existe:", destino); return
    os.makedirs(os.path.dirname(destino) or ".", exist_ok=True)
    print("descargando:", destino)
    req = urllib.request.Request(url, headers={"User-Agent": "pruebas-comedor/1.0"})
    with urllib.request.urlopen(req) as r, open(destino + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    os.replace(destino + ".part", destino)


for rel, url in ARCHIVOS.items():
    bajar(url, os.path.join(DEST, rel))

# LFW funneled: se usa el descargador de scikit-learn (solo baja y descomprime, no carga en memoria)
from sklearn.datasets._lfw import _check_fetch_lfw
ruta, _ = _check_fetch_lfw(data_home=os.path.join(DEST, "lfw"), funneled=True, download_if_missing=True)
print("LFW listo en:", ruta)
print("Listo. Siguiente paso: python scripts/pruebas/embeddings_poblacion.py")
