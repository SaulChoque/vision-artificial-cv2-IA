import base64
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from proyecto import procesar_imagen


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

load_dotenv(BASE_DIR.parent / ".env")

APP_TITLE = os.getenv("APP_TITLE", "Mini Proyecto de Visión Artificial")
APP_DESCRIPTION = os.getenv(
    "APP_DESCRIPTION",
    "Detector de bordes Sobel/Canny para exposición (FastAPI + OpenCV).",
)
APP_URL = os.getenv("APP_URL", "http://127.0.0.1:8000")
CANVA_PRESENTATION_URL = os.getenv("CANVA_PRESENTATION_URL", "")
SOCIAL_PROFILE_URL = os.getenv("SOCIAL_PROFILE_URL", "https://x.com/sebamorido")


app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _img_to_base64_png(img: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la imagen de salida")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_title": APP_TITLE,
            "app_description": APP_DESCRIPTION,
            "app_url": APP_URL,
            "canva_presentation_url": CANVA_PRESENTATION_URL,
        },
    )


@app.post("/api/procesar")
async def procesar(
    archivo: UploadFile = File(...),
    umbral: int = Form(200),
    escala: float = Form(1.0),
):
    contenido = await archivo.read()
    if not contenido:
        raise HTTPException(status_code=400, detail="No se recibió ninguna imagen")

    np_img = np.frombuffer(contenido, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Formato de imagen inválido")

    if umbral < 1 or umbral > 255:
        raise HTTPException(status_code=400, detail="El umbral debe estar entre 1 y 255")

    if escala <= 0 or escala > 4:
        raise HTTPException(status_code=400, detail="La escala debe estar entre 0 y 4")

    try:
        salida = procesar_imagen(img, umbral=umbral, escala=escala)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {exc}") from exc

    return JSONResponse(
        {
            "parametros": salida["parametros"],
            "imagenes": {
                "sobel": _img_to_base64_png(salida["sobel"]),
                "canny": _img_to_base64_png(salida["canny"]),
            },
        }
    )
