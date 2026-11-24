from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.routes import registro, auth, reportes, rafaga, calibracion
from app.services.cache_service import EmbeddingCache
from app.core.database import AsyncSessionLocal
from loguru import logger

app = FastAPI(title="Sistema Comedor UNPRG")
templates = Jinja2Templates(directory="templates")

# Routers
app.include_router(registro.router)
app.include_router(auth.router)
app.include_router(reportes.router)
app.include_router(rafaga.router)
app.include_router(calibracion.router)


@app.on_event("startup")
async def startup_event():
    logger.info("Inicializando sistema...")

    # Cargar cache de embeddings en memoria
    async with AsyncSessionLocal() as db:
        cache = EmbeddingCache.get_instance()
        await cache.load(db)

    logger.info(f"Sistema listo - {cache.count} comensales en cache")


# --- Paginas HTML ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/registro", response_class=HTMLResponse)
async def registro_view(request: Request):
    return templates.TemplateResponse("registro.html", {"request": request})


@app.get("/reconocimiento")
async def reconocimiento_view():
    # El kiosco es la pagina principal; se mantiene la ruta vieja por enlaces guardados
    return RedirectResponse("/")


@app.get("/login", response_class=HTMLResponse)
async def login_view(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
