from fastapi import APIRouter, Depends, HTTPException, Request, status
from typing import List, Dict, Any, Optional
import pandas as pd
import time
from pydantic import BaseModel, ValidationError
import numpy as np
import logging  # Importar el módulo de logging
# Importar erf para el percentil del ejemplo de producción
from fastapi import Body
from scipy.special import erf

from app.models.ExamScorePredictModel import ExamScorePredictModel
from app.schemas.exam_score_data import (
    StudentScoreData, BatchStudentScoreData, ScorePredictionResponse,
    ScorePredictionWithExplanation, ScorePredictionWithRecommendations,
    BatchScorePredictionResponse, BatchScorePredictionResult,
    BatchScorePredictionStats, ScoreCategoryDistribution, ScoreCategory,
    ScoreFeatureImportance, ScoreExplanationData, ScoreRecommendationItem
)

# Configurar logger (similar al ejemplo de producción)
logger = logging.getLogger(__name__)

router = APIRouter()


def get_model(request: Request) -> ExamScorePredictModel:
    """
    Dependencia para obtener el modelo de predicción de exámenes
    que fue cargado al inicio de la aplicación.
    """
    if not hasattr(request.app.state, "models") or "exam_model" not in request.app.state.models or request.app.state.models["exam_model"] is None:
        logger.error(
            "El modelo de predicción de exámenes no está cargado en el estado de la aplicación")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio de predicción no disponible. El modelo no se ha cargado correctamente."
        )
    return request.app.state.models["exam_model"]


def categorize_score(score: float) -> ScoreCategory:
    """Categoriza una puntuación según rangos predefinidos."""
    if score >= 85:
        return ScoreCategory.EXCELENTE
    elif score >= 75:
        return ScoreCategory.BUENO
    elif score >= 65:
        return ScoreCategory.SATISFACTORIO
    elif score >= 55:
        return ScoreCategory.NECESITA_MEJORA
    else:
        return ScoreCategory.INSUFICIENTE


def calculate_percentile(score: float) -> int:
    """
    Calcula el percentil aproximado basado en distribución normal.
    Adaptado para usar la función erf como en el ejemplo de producción.
    """
    mean_score = 70  # Media estimada
    std_dev = 10   # Desviación estándar estimada

    percentile = int(
        100 * (1 - 0.5 * (1 + erf((mean_score - score) / (std_dev * np.sqrt(2))))))
    return max(0, min(100, percentile))


def get_key_factors(model: ExamScorePredictModel, student_data: Dict[str, Any]) -> List[str]:
    """
    Identifica los factores clave que influyen en la predicción para un estudiante específico.
    """
    importance_df = model.get_feature_importance().head(5)
    key_features = importance_df['feature'].tolist()
    factors = []

    for feature in key_features:
        if feature == "pretest":
            factors.append(
                f"Puntuación en prueba previa: {student_data.get('pretest', 'N/A')}")
        elif "teaching_method" in feature:
            factors.append(
                f"Método de enseñanza: {student_data.get('teaching_method', 'N/A')}")
        elif "school_setting" in feature:
            factors.append(
                f"Entorno escolar: {student_data.get('school_setting', 'N/A')}")
        elif "school_type" in feature:
            factors.append(
                f"Tipo de escuela: {student_data.get('school_type', 'N/A')}")
        elif "n_student" == feature:
            factors.append(
                f"Número de estudiantes en el aula: {student_data.get('n_student', 'N/A')}")

    return factors[:3]


def generate_score_explanation(score: float, percentile: int, key_factors: List[str]) -> ScoreExplanationData:
    """
    Genera una explicación para una puntuación predicha.
    """
    category = categorize_score(score)

    summaries = {
        ScoreCategory.EXCELENTE: "El estudiante muestra un desempeño sobresaliente, superando considerablemente las expectativas académicas.",
        ScoreCategory.BUENO: "El estudiante muestra un buen desempeño, superando las expectativas académicas.",
        ScoreCategory.SATISFACTORIO: "El estudiante muestra un desempeño satisfactorio, cumpliendo con las expectativas académicas.",
        ScoreCategory.NECESITA_MEJORA: "El estudiante necesita mejorar en algunas áreas para alcanzar las expectativas académicas.",
        ScoreCategory.INSUFICIENTE: "El estudiante muestra dificultades significativas y requiere apoyo adicional inmediato."
    }

    percentile_text = f"El estudiante se encuentra en el percentil {percentile}, "
    if percentile > 90:
        percentile_text += "superando al 90% de sus compañeros."
    elif percentile > 75:
        percentile_text += "superando a la mayoría de sus compañeros."
    elif percentile > 50:
        percentile_text += "superando a más de la mitad de sus compañeros."
    elif percentile > 25:
        percentile_text += "por debajo del promedio pero por encima del 25% inferior."
    else:
        percentile_text += "ubicándose en el 25% inferior en comparación con sus compañeros."

    return ScoreExplanationData(
        summary=summaries[category],
        percentile=percentile_text,
        key_factors=key_factors
    )


def generate_recommendations(score: float, student_data: Dict[str, Any]) -> List[ScoreRecommendationItem]:
    """
    Genera recomendaciones personalizadas basadas en la puntuación predicha y datos del estudiante.
    """
    category = categorize_score(score)
    recommendations = []

    if category in [ScoreCategory.INSUFICIENTE, ScoreCategory.NECESITA_MEJORA]:
        recommendations.append(
            ScoreRecommendationItem(
                area="Tutoría académica",
                recommendation="Se recomienda inscribir al estudiante en el programa de tutoría académica para recibir apoyo adicional en las materias principales."
            )
        )
        recommendations.append(
            ScoreRecommendationItem(
                area="Seguimiento individual",
                recommendation="Implementar un plan de seguimiento semanal para monitorear el progreso y ajustar las intervenciones según sea necesario."
            )
        )

    if category == ScoreCategory.SATISFACTORIO:
        recommendations.append(
            ScoreRecommendationItem(
                area="Refuerzo de habilidades",
                recommendation="Fortalecer las áreas donde el estudiante muestra potencial de mejora mediante actividades adicionales de práctica."
            )
        )

    if category in [ScoreCategory.BUENO, ScoreCategory.EXCELENTE]:
        recommendations.append(
            ScoreRecommendationItem(
                area="Enriquecimiento académico",
                recommendation="Ofrecer oportunidades de aprendizaje avanzado para mantener el interés y motivación del estudiante."
            )
        )

    pretest = student_data.get("pretest", 0)
    if pretest < 50:
        recommendations.append(
            ScoreRecommendationItem(
                area="Nivelación académica",
                recommendation="Implementar un programa de nivelación para fortalecer los conocimientos básicos antes de avanzar a temas más complejos."
            )
        )

    if student_data.get("lunch") == "Qualifies for reduced/free lunch":
        recommendations.append(
            ScoreRecommendationItem(
                area="Recursos educativos",
                recommendation="Proporcionar acceso a recursos educativos adicionales que el estudiante pueda utilizar fuera del horario escolar."
            )
        )

    return recommendations[:4]


@router.post(
    "/",
    response_model=ScorePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predicción individual de puntuación",
    description="Realiza una predicción de puntuación para un estudiante individual"
)
async def predict_student_score(
    request: Request,  # Añadimos request para acceder al estado de la aplicación
    # Documentación en Body
    student_data: StudentScoreData = Body(
        ..., description="Datos del estudiante para la predicción"),
    model: ExamScorePredictModel = Depends(get_model)
):
    """
    Endpoint para predecir la puntuación de examen para un estudiante individual.
    """
    start_time = time.time()
    # Log al inicio
    logger.info(
        f"Inicio de predicción individual para estudiante con datos: {student_data}")

    try:
        student_dict = student_data.model_dump()
        prediction = model.predict(student_dict)
        top_features = model.get_feature_importance().head(5).to_dict(orient="records")
        processing_time = int((time.time() - start_time) * 1000)

        response = ScorePredictionResponse(
            prediction=prediction,
            processing_time_ms=processing_time,
            top_features=[ScoreFeatureImportance(
                **feature) for feature in top_features]
        )
        # Log al finalizar con éxito
        logger.info(
            f"Predicción individual exitosa para estudiante, tiempo de procesamiento: {processing_time}ms")
        return response

    except ValidationError as ve:  # Capturar errores de validación de Pydantic
        # Log de warning en validación
        logger.warning(
            f"Error de validación en la predicción individual: {str(ve)}")
        raise HTTPException(
            # Código de error 422 para validación
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Datos inválidos: {str(ve)}"
        )
    except Exception as e:  # Capturar cualquier otra excepción
        # Log detallado del error
        logger.error(
            f"Error al realizar la predicción individual: {str(e)}", exc_info=True)
        raise HTTPException(
            # Código de error 500 para errores internos
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al realizar la predicción: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchScorePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predicción en lote de puntuaciones",
    description="Predice las puntuaciones de examen para múltiples estudiantes en lote"
)
async def predict_batch_student_scores(
    request: Request,  # Añadimos request para acceder al estado de la aplicación
    # Documentación en Body
    batch_data: BatchStudentScoreData = Body(
        ..., description="Datos de lote de estudiantes para la predicción"),
    model: ExamScorePredictModel = Depends(get_model)
):
    """
    Endpoint para predecir las puntuaciones de examen para múltiples estudiantes en lote.
    """
    start_time = time.time()
    # Log al inicio del batch
    logger.info(
        f"Inicio de predicción en lote para {len(batch_data.students)} estudiantes")

    try:
        results = []
        predictions = []
        student_dicts = [student.model_dump()
                         for student in batch_data.students]
        df = pd.DataFrame(student_dicts)
        batch_predictions = model.predict(df)

        for i, (prediction, student_data) in enumerate(zip(batch_predictions, student_dicts)):
            category = categorize_score(prediction)
            predictions.append(prediction)
            results.append(
                BatchScorePredictionResult(
                    student_index=i,
                    student_id=student_data.get("student_id"),
                    prediction=prediction,
                    category=category,
                    percentile=calculate_percentile(prediction),
                    student_data=student_data
                )
            )

        category_counts = {}
        for result in results:
            category = result.category
            category_counts[category] = category_counts.get(category, 0) + 1

        category_distribution = [
            ScoreCategoryDistribution(
                category=category,
                count=count,
                percentage=round(count * 100 / len(results), 1)
            )
            for category, count in category_counts.items()
        ]
        category_distribution.sort(key=lambda x: str(x.category))

        stats = BatchScorePredictionStats(
            count=len(results),
            average_prediction=sum(predictions) /
            len(predictions) if predictions else None,
            min_prediction=min(predictions) if predictions else None,
            max_prediction=max(predictions) if predictions else None,
            category_distribution=category_distribution,
            processing_time_ms=int((time.time() - start_time) * 1000)
        )

        response = BatchScorePredictionResponse(
            results=results,
            stats=stats
        )
        # Log al finalizar batch
        logger.info(
            f"Predicción en lote exitosa para {len(batch_data.students)} estudiantes, tiempo de procesamiento: {stats.processing_time_ms}ms")
        return response

    except ValidationError as ve:  # Capturar errores de validación
        # Log de warning en validación batch
        logger.warning(
            f"Error de validación en la predicción en lote: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # Código de error 422
            detail=f"Datos inválidos en lote: {str(ve)}"
        )
    except Exception as e:  # Capturar otras excepciones
        # Log detallado de error batch
        logger.error(
            f"Error al realizar las predicciones en lote: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,  # Código 500 para error interno
            detail=f"Error interno al realizar las predicciones en lote: {str(e)}"
        )


@router.post(
    "/explain",
    response_model=ScorePredictionWithExplanation,
    status_code=status.HTTP_200_OK,
    summary="Predicción con explicación detallada",
    description="Predice la puntuación de un estudiante y proporciona una explicación detallada"
)
async def predict_with_explanation(
    request: Request,  # Añadimos request para acceder al estado de la aplicación
    # Documentación en Body
    student_data: StudentScoreData = Body(
        ..., description="Datos del estudiante para la predicción con explicación"),
    model: ExamScorePredictModel = Depends(get_model)
):
    """
    Endpoint para predecir la puntuación de un estudiante y proporcionar una explicación detallada.
    """
    start_time = time.time()
    # Log al inicio de explicación
    logger.info(
        f"Inicio de predicción con explicación para estudiante con datos: {student_data}")

    try:
        student_dict = student_data.model_dump()
        prediction = model.predict(student_dict)
        category = categorize_score(prediction)
        percentile = calculate_percentile(prediction)
        importance_df = model.get_feature_importance()
        top_features = [ScoreFeatureImportance(
            **feature) for feature in importance_df.head(5).to_dict(orient="records")]
        key_factors = get_key_factors(model, student_dict)
        explanation = generate_score_explanation(
            prediction, percentile, key_factors)
        processing_time = int((time.time() - start_time) * 1000)

        response = ScorePredictionWithExplanation(
            prediction=prediction,
            category=category,
            percentile=percentile,
            explanation=explanation,
            top_features=top_features,
            processing_time_ms=processing_time
        )
        # Log de éxito en explicación
        logger.info(
            f"Predicción con explicación exitosa para estudiante, tiempo de procesamiento: {processing_time}ms")
        return response

    except ValidationError as ve:  # Capturar errores de validación
        # Log warning en validación explicación
        logger.warning(
            f"Error de validación en la predicción con explicación: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # Código 422
            detail=f"Datos inválidos: {str(ve)}"
        )
    except Exception as e:  # Capturar otras excepciones
        # Log detallado error explicación
        logger.error(
            f"Error al realizar la predicción con explicación: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,  # Código 500
            detail=f"Error interno al realizar la predicción con explicación: {str(e)}"
        )


@router.post(
    "/recommend",
    response_model=ScorePredictionWithRecommendations,
    status_code=status.HTTP_200_OK,
    summary="Predicción con recomendaciones personalizadas",
    description="Predice la puntuación y proporciona recomendaciones personalizadas para el estudiante"
)
async def predict_with_recommendations(
    request: Request,  # Añadimos request para acceder al estado de la aplicación
    # Documentación en Body
    student_data: StudentScoreData = Body(
        ..., description="Datos del estudiante para la predicción con recomendaciones"),
    model: ExamScorePredictModel = Depends(get_model)
):
    """
    Endpoint para predecir la puntuación de un estudiante y proporcionar recomendaciones personalizadas.
    """
    start_time = time.time()
    # Log al inicio de recomendaciones
    logger.info(
        f"Inicio de predicción con recomendaciones para estudiante con datos: {student_data}")

    try:
        student_dict = student_data.model_dump()
        prediction = model.predict(student_dict)
        category = categorize_score(prediction)
        percentile = calculate_percentile(prediction)
        recommendations = generate_recommendations(prediction, student_dict)
        processing_time = int((time.time() - start_time) * 1000)

        response = ScorePredictionWithRecommendations(
            student_id=student_dict.get("student_id"),
            prediction=prediction,
            category=category,
            percentile=percentile,
            recommendations=recommendations,
            student_data=student_dict,
            processing_time_ms=processing_time
        )
        # Log de éxito en recomendaciones
        logger.info(
            f"Predicción con recomendaciones exitosa para estudiante, tiempo de procesamiento: {processing_time}ms")
        return response

    except ValidationError as ve:  # Capturar errores de validación
        # Log warning en validación recomendaciones
        logger.warning(
            f"Error de validación en la predicción con recomendaciones: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # Código 422
            detail=f"Datos inválidos: {str(ve)}"
        )
    except Exception as e:  # Capturar otras excepciones
        # Log detallado error recomendaciones
        logger.error(
            f"Error al realizar la predicción con recomendaciones: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,  # Código 500
            detail=f"Error interno al realizar la predicción con recomendaciones: {str(e)}"
        )


# Función para cargar el modelo al inicio de la aplicación (puedes colocar esto en tu archivo principal app.py o main.py)
async def load_model_to_state(app):
    """
    Carga el modelo de predicción al estado de la aplicación al inicio.
    """
    try:
        model = ExamScorePredictModel.load_from_file()
        app.state.exam_model = model
        # Log de carga exitosa
        logger.info(
            "Modelo de predicción de exámenes cargado exitosamente al estado de la aplicación.")
    except FileNotFoundError:
        # Log de error si no se encuentra el archivo
        logger.error(
            "Archivo del modelo de predicción no encontrado.", exc_info=True)
        # Asegurar que el modelo en estado sea None en caso de error
        app.state.exam_model = None
    except Exception as e:
        # Log detallado de error de carga
        logger.error(
            f"Error al cargar el modelo de predicción al estado de la aplicación: {str(e)}", exc_info=True)
        # Asegurar que el modelo en estado sea None en caso de error
        app.state.exam_model = None
