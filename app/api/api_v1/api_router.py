# app/api/api_v1/api_router.py
from fastapi import APIRouter
from app.api.api_v1.endpoints.performance import predictions as performance_predictions
from app.api.api_v1.endpoints.score import predictions as score_predictions
# Importaciones para modelos futuros
# from app.api.api_v1.endpoints.otro_modelo import predictions as otro_predictions

api_router = APIRouter()

# Endpoints para el modelo de rendimiento estudiantil
api_router.include_router(
    performance_predictions.router,
    prefix="/performance/predictions",
    tags=["Performance - Predicciones"]
)
api_router.include_router(
    score_predictions.router,
    prefix="/score/predictions",
    tags=["Score - Predicciones"]
)

# Espacio para endpoints de modelos futuros
# api_router.include_router(
#     otro_predictions.router,
#     prefix="/otro-modelo/predictions",
#     tags=["Otro Modelo - Predicciones"]
# )
