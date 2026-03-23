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

## Variables de entorno

- `APP_TITLE`
- `APP_DESCRIPTION`
- `APP_URL`
- `SOCIAL_PROFILE_URL`
- `CANVA_PRESENTATION_URL`

## Deploy en Vercel

1. Asegúrate de tener `requirements.txt` y `vercel.json`.
2. En Vercel, importa este repositorio/carpeta.
3. Configura variables de entorno del archivo `env.example`.
4. Deploy.

La entrada serverless está en `api/index.py`.

## Nota sobre OpenCL

`proyecto.py` intenta usar PyOpenCL si está disponible. En entornos serverless (como Vercel) normalmente no hay GPU/OpenCL, por lo que la app usa fallback CPU automáticamente.
