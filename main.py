from app.api.api_v1.api_router import api_router
import os
import uvicorn
import logging
import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from contextlib import asynccontextmanager
from app.core.config import settings
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Validar configuración necesaria


def validate_settings():
    required_settings = ["API_V1_STR", "PROJECT_NAME", "VERSION"]
    missing = [setting for setting in required_settings if not getattr(
        settings, setting, None)]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Configuración incompleta. Faltan los siguientes valores: {missing_str}")

# Función para cargar y preparar recursos al iniciar la aplicación


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Iniciando {settings.PROJECT_NAME} v{settings.VERSION}")

    try:
        # Validar configuración
        validate_settings()
        logger.info("Configuración validada correctamente")

        app.state.models = {}  # Diccionario para almacenar los modelos

        # Cargar o entrenar modelo de StudentPerformanceFactorModel
        from app.models.StudentPerformanceModel import StudentPerformanceModel
        try:
            logger.info(
                f"Intentando cargar performance model desde: {settings.PERFORMANCE_MODEL_PATH}")
            performance_model = StudentPerformanceModel.load_from_file(
                settings.PERFORMANCE_MODEL_PATH)
            logger.info("Performance model cargado exitosamente")
        except FileNotFoundError:
            logger.warning(
                f"Performance model no encontrado en {settings.PERFORMANCE_MODEL_PATH}, se entrenará uno nuevo")
            csv_path = os.path.join(
                settings.CSV_PATH, 'StudentPerformanceFactors.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"No se encontró el archivo de datos: {csv_path}")
            df = pd.read_csv(csv_path)
            performance_model, evaluation = StudentPerformanceModel.train_from_dataframe(
                df)
            saved_path = performance_model.save_model(
                settings.PERFORMANCE_MODEL_PATH)
            logger.info(
                f"Performance model entrenado y guardado en: {saved_path}")

        app.state.models["performance_model"] = performance_model

        # Cargar o entrenar modelo de ExamScorePredictModel
        from app.models.ExamScorePredictModel import ExamScorePredictModel
        try:
            logger.info(
                f"Intentando cargar exam model desde: {settings.EXAM_MODEL_PATH}")
            exam_model = ExamScorePredictModel.load_from_file(
                settings.EXAM_MODEL_PATH)
            logger.info("Exam model cargado exitosamente")
        except FileNotFoundError:
            logger.warning(
                f"Exam model no encontrado en {settings.EXAM_MODEL_PATH}, se entrenará uno nuevo")
            csv_path = os.path.join(settings.CSV_PATH, 'ExamScore.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"No se encontró el archivo de datos: {csv_path}")
            df = pd.read_csv(csv_path)
            exam_model, evaluation = ExamScorePredictModel.train_from_dataframe(
                df)
            saved_path = exam_model.save_model(settings.EXAM_MODEL_PATH)
            logger.info(f"Exam model entrenado y guardado en: {saved_path}")

        app.state.models["exam_model"] = exam_model

    except Exception as e:
        logger.error(
            f"Error fatal durante la inicialización: {str(e)}", exc_info=True)
        raise

    yield

    logger.info("Liberando recursos...")


# Crear la aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan
)

# Configurar CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin)
                       for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Manejadores de excepciones para respuestas consistentes


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Manejador personalizado para errores de validación.
    Formatea el error para que sea más legible y asegura que sea serializable a JSON.
    """
    logger.warning(f"Validación fallida: {exc.errors()}")

    # Crear una versión serializable de los errores
    formatted_errors = []
    for error in exc.errors():
        formatted_error = {
            "type": error.get("type", ""),
            "loc": " > ".join([str(loc) for loc in error.get("loc", [])]),
            "msg": error.get("msg", ""),
            "input": error.get("input", "")
        }
        # Evitar incluir el objeto ValueError directamente
        if "ctx" in error and isinstance(error["ctx"], dict) and "error" in error["ctx"]:
            # Si ctx.error es un objeto de excepción, solo incluir su mensaje como string
            if isinstance(error["ctx"]["error"], Exception):
                formatted_error["detail"] = str(error["ctx"]["error"])
            else:
                formatted_error["detail"] = error["ctx"]["error"]

        formatted_errors.append(formatted_error)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": formatted_errors}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error no manejado: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor"}
    )

# Importar routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Endpoint raíz que proporciona información básica sobre la API"""
    return {
        "message": "API de Predicción de Rendimiento Estudiantil",
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de salud de la API"""
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model") and app.state.model is not None
    }

if __name__ == "__main__":
    try:
        logger.info(f"Iniciando servidor en http://0.0.0.0:8000")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.critical(
            f"Error al iniciar el servidor: {str(e)}", exc_info=True)
        sys.exit(1)
