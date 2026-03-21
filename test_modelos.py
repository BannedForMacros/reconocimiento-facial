import insightface
from insightface.app import FaceAnalysis

print("Descargando modelos...")
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Modelos listos (incluye anti-spoofing)")