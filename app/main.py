from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.routes import registro, reconocimiento, auth, reportes

app = FastAPI(title="Sistema Comedor UNPRG")
templates = Jinja2Templates(directory="templates")

# Routers
app.include_router(registro.router)
app.include_router(reconocimiento.router)
app.include_router(auth.router)
app.include_router(reportes.router)

# Páginas HTML
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/registro", response_class=HTMLResponse)
async def registro_view(request: Request):
    return templates.TemplateResponse("registro.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_view(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)