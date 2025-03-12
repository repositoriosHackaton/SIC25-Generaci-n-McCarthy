# app/core/config.py
from pydantic_settings import BaseSettings
import os
from typing import List, Optional


class Settings(BaseSettings):
    """Configuraciones de la aplicación"""
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "API de Predicción de Rendimiento Estudiantil"
    DESCRIPTION: str = "API para identificar patrones de rendimiento académico y generar predicciones"
    VERSION: str = "0.1.0"

    # Configuración de seguridad
    SECRET_KEY: str = os.getenv("SECRET_KEY", "cambiar_en_produccion")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 días

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # Rutas de archivos
    EXAM_MODEL_PATH: str = os.getenv(
        "MODEL_PATH", "app/models/trained/exam_score_predict_model.joblib")
    PERFORMANCE_MODEL_PATH: str = os.getenv(
        "MODEL_PATH", "app/models/trained/student_performance_model.joblib")

    CSV_PATH: str = os.getenv(
        "CSV_PATH", "app/db/")

    # Límites de la API
    MAX_BATCH_SIZE: int = 100

    class Config:
        case_sensitive = False


settings = Settings()
