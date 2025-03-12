# app/api/api_v1/endpoints/performance/predictions.py

from fastapi import Depends, Request, HTTPException, status
from fastapi import APIRouter, HTTPException, Request, status, Body, Query, Path, Depends
from app.models.StudentPerformanceModel import StudentPerformanceModel
from app.schemas.student_performance_data import (
    StudentPerformanceData,
    BatchStudentData,
    PredictionResponse,
    BatchPredictionResponse,
    PredictionWithExplanation,
    PredictionWithRecommendations
)
from typing import List, Optional, Dict, Any
import logging
from pydantic import ValidationError
import pandas as pd
import time
import numpy as np
from scipy.special import erf


# Configurar logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Función auxiliar para validar modelo


def get_model(request: Request) -> StudentPerformanceModel:
    """
    Dependencia para obtener el modelo de predicción de exámenes
    que fue cargado al inicio de la aplicación.
    """
    if not hasattr(request.app.state, "models") or "performance_model" not in request.app.state.models or request.app.state.models["performance_model"] is None:
        logger.error(
            "El modelo de predicción de exámenes no está cargado en el estado de la aplicación")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio de predicción no disponible. El modelo no se ha cargado correctamente."
        )
    return request.app.state.models["performance_model"]

# Funciones auxiliares para interpretar predicciones


def interpret_score(score: float) -> Dict[str, Any]:
    """Interpreta el puntaje de examen y devuelve categorías y recomendaciones."""
    performance_categories = {
        "Excelente": (90, 100),
        "Bueno": (80, 89.99),
        "Satisfactorio": (70, 79.99),
        "Necesita mejorar": (60, 69.99),
        "En riesgo": (0, 59.99)
    }

    # Determinar categoría
    category = "Desconocido"
    for cat, (min_val, max_val) in performance_categories.items():
        if min_val <= score <= max_val:
            category = cat
            break

    return {
        "score": score,
        "category": category,
        "percentile": get_percentile(score)
    }


def get_percentile(score: float) -> int:
    """
    Calcula el percentil aproximado del puntaje.
    En un sistema real, esto se calcularía basado en datos históricos.
    """
    # Distribución simplificada - esto debería basarse en datos reales
    mean_score = 75
    std_dev = 15

    # Aproximación de percentil usando distribución normal
    percentile = int(
        100 * (1 - 0.5 * (1 + erf((mean_score - score) / (std_dev * np.sqrt(2))))))
    return max(0, min(100, percentile))


def generate_recommendations(student_data: Dict[str, Any], score: float, feature_importance: pd.DataFrame = None) -> List[Dict[str, str]]:
    """
    Genera recomendaciones personalizadas basadas en los datos del estudiante,
    la predicción y la importancia de características.
    """
    recommendations = []

    # Recomendaciones basadas en el puntaje predicho
    if score < 60:
        recommendations.append({
            "area": "Académico",
            "recommendation": "Se recomienda inscribirse en programas de tutorías adicionales urgentemente."
        })
    elif score < 70:
        recommendations.append({
            "area": "Académico",
            "recommendation": "Considerar aumentar las horas de estudio y buscar apoyo académico."
        })

    # Recomendaciones basadas en factores específicos
    # Horas de estudio
    hours_studied = student_data.get("Hours_Studied", 0)
    if hours_studied < 5:
        recommendations.append({
            "area": "Hábitos de estudio",
            "recommendation": "Aumentar las horas de estudio semanales. Se recomienda un mínimo de 8 horas."
        })

    # Asistencia
    attendance = student_data.get("Attendance", 0)
    if attendance < 80:
        recommendations.append({
            "area": "Asistencia",
            "recommendation": "Mejorar la asistencia a clases para no perder contenido importante."
        })

    # Horas de sueño
    sleep_hours = student_data.get("Sleep_Hours", 0)
    if sleep_hours < 7:
        recommendations.append({
            "area": "Bienestar",
            "recommendation": "Dormir al menos 7-8 horas para mejorar la concentración y rendimiento."
        })

    # Recomendaciones basadas en importancia de características
    if feature_importance is not None and not feature_importance.empty:
        # Obtener la característica más importante
        top_feature = feature_importance.iloc[0]["feature"]
        if "Hours_Studied" in top_feature:
            recommendations.append({
                "area": "Prioridad",
                "recommendation": "Las horas de estudio tienen el mayor impacto en tu rendimiento. Considera incrementarlas."
            })
        elif "Attendance" in top_feature:
            recommendations.append({
                "area": "Prioridad",
                "recommendation": "La asistencia a clases tiene un gran impacto en tu rendimiento. Mejora tu asistencia."
            })

    return recommendations


@router.post(
    "/",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predicción individual de rendimiento",
    description="Realiza una predicción de rendimiento estudiantil para un solo estudiante"
)
async def predict_performance(
    data: StudentPerformanceData = Body(...),
    include_features: bool = Query(
        False, description="Incluir información sobre características importantes"),
    model: StudentPerformanceModel = Depends(get_model)
):
    """
    Realiza una predicción básica de rendimiento estudiantil para un estudiante.

    - **data**: Datos del estudiante para la predicción
    - **include_features**: Si es True, incluye información sobre las características más importantes
    """
    start_time = time.time()

    try:
        # Preprocesar datos
        processed_data = model.preprocess_data(data.model_dump())

        # Realizar predicción
        prediction = model.predict(processed_data)

        # Construir respuesta
        response = {
            "prediction": float(prediction),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

        # Incluir información de características si se solicita
        if include_features:
            importance = model.get_feature_importance()
            top_features = importance.head(5).to_dict(orient="records")
            response["top_features"] = top_features

        logger.info(f"Predicción exitosa: {prediction:.2f}")
        return response

    except ValidationError as ve:
        logger.warning(f"Error de validación: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Datos inválidos: {str(ve)}"
        )
    except Exception as e:
        logger.error(
            f"Error al realizar la predicción: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar la predicción: {str(e)}"
        )


@router.post(
    "/explain",
    response_model=PredictionWithExplanation,
    status_code=status.HTTP_200_OK,
    summary="Predicción con explicación",
    description="Realiza una predicción y proporciona una explicación detallada"
)
async def predict_with_explanation(
    data: StudentPerformanceData = Body(...),
    model: StudentPerformanceModel = Depends(get_model)
):
    """
    Realiza una predicción de rendimiento estudiantil y proporciona una explicación
    detallada de la misma en lenguaje natural.

    - **data**: Datos del estudiante para la predicción
    """
    start_time = time.time()

    try:
        # Preprocesar datos
        processed_data = model.preprocess_data(data.model_dump())

        # Realizar predicción
        prediction = model.predict(processed_data)

        # Obtener importancia de características
        feature_importance = model.get_feature_importance()
        top_features = feature_importance.head(5).to_dict(orient="records")

        # Interpretar el puntaje
        interpretation = interpret_score(prediction)

        # Construir respuesta con explicación
        explanation = {
            "summary": f"El estudiante obtendrá aproximadamente {prediction:.1f} puntos en el examen, lo cual se considera un rendimiento '{interpretation['category']}'.",
            "percentile": f"Este puntaje está por encima del {interpretation['percentile']}% de los estudiantes.",
            "key_factors": []
        }

        # Añadir factores clave que afectan el rendimiento
        for idx, feature in enumerate(top_features):
            feature_name = feature["feature"]
            coefficient = feature["coefficient"]

            # Simplificar el nombre de la característica
            simplified_name = feature_name.replace("_", " ").title()

            # Determinar si es positivo o negativo
            impact = "positivo" if coefficient > 0 else "negativo"
            explanation["key_factors"].append(
                f"{simplified_name} tiene un impacto {impact} significativo en el rendimiento (posición {idx+1} en importancia)."
            )

        response = {
            "prediction": float(prediction),
            "category": interpretation["category"],
            "percentile": interpretation["percentile"],
            "explanation": explanation,
            "top_features": top_features,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

        logger.info(f"Predicción con explicación exitosa: {prediction:.2f}")
        return response

    except Exception as e:
        logger.error(
            f"Error al realizar la predicción con explicación: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar la explicación: {str(e)}"
        )


@router.post(
    "/recommend",
    response_model=PredictionWithRecommendations,
    status_code=status.HTTP_200_OK,
    summary="Predicción con recomendaciones",
    description="Realiza una predicción y proporciona recomendaciones personalizadas"
)
async def predict_with_recommendations(
    data: StudentPerformanceData = Body(...),
    model: StudentPerformanceModel = Depends(get_model)
):
    """
    Realiza una predicción de rendimiento estudiantil y proporciona recomendaciones
    personalizadas para mejorar el rendimiento.

    - **data**: Datos del estudiante para la predicción
    """
    start_time = time.time()

    try:
        # Preprocesar datos
        processed_data = model.preprocess_data(data.model_dump())

        # Realizar predicción
        prediction = model.predict(processed_data)

        # Obtener importancia de características
        feature_importance = model.get_feature_importance()

        # Interpretar el puntaje
        interpretation = interpret_score(prediction)

        # Generar recomendaciones personalizadas
        recommendations = generate_recommendations(
            processed_data,
            prediction,
            feature_importance
        )

        response = {
            "prediction": float(prediction),
            "category": interpretation["category"],
            "percentile": interpretation["percentile"],
            "recommendations": recommendations,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

        logger.info(
            f"Predicción con recomendaciones exitosa: {prediction:.2f}")
        return response

    except Exception as e:
        logger.error(
            f"Error al realizar la predicción con recomendaciones: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar recomendaciones: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predicción en lote",
    description="Realiza predicciones en lote para múltiples estudiantes"
)
async def predict_performance_batch(
    batch_data: BatchStudentData = Body(...),
    detailed_results: bool = Query(
        False, description="Incluir resultados detallados"),
    include_categories: bool = Query(
        True, description="Incluir categorías de rendimiento"),
    max_batch_size: int = Query(
        100, description="Tamaño máximo del lote permitido"),
    model: StudentPerformanceModel = Depends(get_model)
):
    """
    Realiza predicciones en lote para múltiples estudiantes.

    - **batch_data**: Lista de datos de estudiantes para realizar predicciones
    - **detailed_results**: Si es True, incluye información más detallada en los resultados
    - **include_categories**: Si es True, incluye categorías de rendimiento para cada predicción
    - **max_batch_size**: Número máximo de estudiantes permitidos en una solicitud
    """
    start_time = time.time()

    try:
        # Validar el tamaño del lote
        if len(batch_data.students) > max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"El tamaño del lote excede el máximo permitido de {max_batch_size} estudiantes"
            )

        # Preprocesar datos
        processed_data_list = [model.preprocess_data(
            student.model_dump()) for student in batch_data.students]

        # Crear DataFrame
        students_df = pd.DataFrame(processed_data_list)

        # Realizar predicciones
        predictions = model.predict(students_df)

        # Construir respuesta
        result = []
        for i, pred in enumerate(predictions):
            student_result = {
                "student_index": i,
                "prediction": float(pred)
            }

            if include_categories:
                interpretation = interpret_score(pred)
                student_result["category"] = interpretation["category"]
                student_result["percentile"] = interpretation["percentile"]

            if detailed_results:
                student_result["student_data"] = processed_data_list[i]

            result.append(student_result)

        # Compilar estadísticas
        stats = {
            "count": len(predictions),
            "average_prediction": float(np.mean(predictions)) if len(predictions) > 0 else None,
            "min_prediction": float(np.min(predictions)) if len(predictions) > 0 else None,
            "max_prediction": float(np.max(predictions)) if len(predictions) > 0 else None,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

        # Añadir distribución por categorías si se solicita
        if include_categories:
            category_counts = {}
            for student_result in result:
                category = student_result.get("category", "Desconocido")
                category_counts[category] = category_counts.get(
                    category, 0) + 1

            stats["category_distribution"] = [
                {"category": cat, "count": count, "percentage": (
                    count / len(predictions)) * 100}
                for cat, count in category_counts.items()
            ]

        logger.info(f"Procesadas {len(predictions)} predicciones en lote")
        return {"results": result, "stats": stats}

    except Exception as e:
        logger.error(
            f"Error al realizar la predicción en lote: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al realizar la predicción en lote: {str(e)}"
        )


@router.post(
    "/by-id/{student_id}",
    response_model=PredictionWithRecommendations,
    status_code=status.HTTP_200_OK,
    summary="Predicción para estudiante específico por ID",
    description="Realiza una predicción con recomendaciones para un estudiante identificado por ID"
)
async def predict_for_student(
    student_id: str = Path(..., description="ID único del estudiante"),
    data: StudentPerformanceData = Body(...,
                                        description="Datos del estudiante"),
    model: StudentPerformanceModel = Depends(get_model)
):
    """
    Realiza una predicción con recomendaciones para un estudiante específico.
    Recibe el ID del estudiante y sus datos completos.

    - **student_id**: ID único del estudiante en el sistema
    - **data**: Datos del estudiante para la predicción
    """
    start_time = time.time()

    try:
        # Preprocesar datos
        processed_data = model.preprocess_data(data.model_dump())

        # Realizar predicción
        prediction = model.predict(processed_data)

        # Obtener importancia de características
        feature_importance = model.get_feature_importance()

        # Interpretar el puntaje
        interpretation = interpret_score(prediction)

        # Generar recomendaciones personalizadas
        recommendations = generate_recommendations(
            processed_data,
            prediction,
            feature_importance
        )

        response = {
            "student_id": student_id,
            "prediction": float(prediction),
            "category": interpretation["category"],
            "percentile": interpretation["percentile"],
            "recommendations": recommendations,
            "student_data": processed_data,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

        logger.info(
            f"Predicción para estudiante {student_id}: {prediction:.2f}")
        return response

    except Exception as e:
        logger.error(
            f"Error al realizar la predicción para estudiante {student_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar predicción: {str(e)}"
        )
