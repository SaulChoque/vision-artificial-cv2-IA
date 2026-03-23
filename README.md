# Mini proyecto FastAPI - Visión Artificial

Aplicación web móvil para cargar o tomar una foto, procesarla con Sobel/Canny (basado en `proyecto.py`) y visualizar resultados con parámetros de análisis.

## Requisitos

- Python 3.11+

## Ejecución local

1. Crear entorno virtual e instalar dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configurar variables de entorno:

```bash
cp env.example .env
```

3. Levantar servidor:

```bash
uvicorn app.main:app --reload
```

4. Abrir:

```text
http://127.0.0.1:8000
```
