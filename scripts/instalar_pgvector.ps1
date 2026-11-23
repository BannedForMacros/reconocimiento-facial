# Compila e instala pgvector 0.8.0 en PostgreSQL 17 (Windows).
# Ejecutar en PowerShell COMO ADMINISTRADOR:
#   powershell -ExecutionPolicy Bypass -File C:\Jesus\reconocimiento-facial\scripts\instalar_pgvector.ps1 [ruta_log]

param([string]$Log = (Join-Path $env:TEMP "instalar_pgvector.log"))

Start-Transcript -Path $Log -Force | Out-Null
# git y nmake escriben avisos en stderr; en PowerShell 5.1 eso no debe cortar el script.
$ErrorActionPreference = "Continue"
$pg  = "C:\Program Files\PostgreSQL\17"
$vc  = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$src = Join-Path $env:TEMP "pgvector-build"

function Fin($msg, $code) { Write-Host $msg; Stop-Transcript | Out-Null; exit $code }

if (-not (Test-Path $vc)) { Fin "ERROR: No se encontro Visual Studio C++ en $vc" 1 }
if (Test-Path $src) { Remove-Item -Recurse -Force $src }

cmd /c "git clone --quiet --depth 1 --branch v0.8.0 https://github.com/pgvector/pgvector.git `"$src`" 2>&1"
if ($LASTEXITCODE -ne 0) { Fin "ERROR: fallo git clone" 1 }

# nmake install copia vector.dll a lib\ y vector.control / vector--*.sql a share\extension\
cmd /c "`"$vc`" >nul && cd /d `"$src`" && set `"PGROOT=$pg`" && nmake /NOLOGO /F Makefile.win && nmake /NOLOGO /F Makefile.win install 2>&1"
if ($LASTEXITCODE -ne 0) { Fin "ERROR: fallo la compilacion o la instalacion de pgvector" 1 }

if (Test-Path "$pg\share\extension\vector.control") {
    Fin "OK: pgvector instalado correctamente en PostgreSQL 17" 0
} else {
    Fin "ERROR: no se encontro vector.control despues de instalar" 1
}
