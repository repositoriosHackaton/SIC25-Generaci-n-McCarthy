from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum


class LunchType(str, Enum):
    """Tipos de almuerzo escolares."""
    QUALIFIES = "Qualifies for reduced/free lunch"
    DOES_NOT_QUALIFY = "Does not qualify"


class TeachingMethod(str, Enum):
    """Métodos de enseñanza."""
    STANDARD = "Standard"
    EXPERIMENTAL = "Experimental"


class SchoolSetting(str, Enum):
    """Entornos escolares."""
    URBAN = "Urban"
    SUBURBAN = "Suburban"
    RURAL = "Rural"


class SchoolType(str, Enum):
    """Tipos de escuela."""
    PUBLIC = "Public"
    NON_PUBLIC = "Non-public"


class Gender(str, Enum):
    """Géneros."""
    MALE = "Male"
    FEMALE = "Female"


class StudentScoreData(BaseModel):
    """Esquema para los datos de entrada de un estudiante para predecir puntaje de examen."""
    # Campos numéricos
    pretest: float = Field(...,
                           description="Puntaje del examen previo", ge=22.0, le=93.0)
    n_student: Optional[float] = Field(
        None, description="Número de estudiantes en el aula", ge=14.0, le=31.0)

    # Campos categóricos
    school: str = Field(...,
                        description="Código de la escuela (p.ej. GOOBU, QOQTS)")
    school_setting: SchoolSetting = Field(..., description="Entorno escolar")
    school_type: SchoolType = Field(..., description="Tipo de escuela")
    classroom: str = Field(..., description="Identificador del aula")
    teaching_method: TeachingMethod = Field(...,
                                            description="Método de enseñanza")
    gender: Gender = Field(..., description="Género")
    lunch: LunchType = Field(..., description="Tipo de almuerzo")

    # Campo opcional para identificación
    student_id: Optional[str] = Field(
        None, description="Identificador del estudiante (opcional)")

    @field_validator('school')
    def validate_school(cls, v):
        # Lista de escuelas válidas basada en los datos
        valid_schools = [
            "GOOBU", "QOQTS", "UKPGS", "DNQDD", "GJJHK", "ZOWMK", "VVTVA", "KZKKE",
            "CCAAW", "CUQAM", "VKWQH", "IDGFP", "UAGPU", "UUUQX", "OJOBU", "CIMBB",
            "ZMNYA", "GOKXL", "LAYPA", "KFZMY", "VHDHF", "FBUMG", "ANKYI"
        ]

        if v not in valid_schools:
            raise ValueError(
                f"El código de escuela proporcionado no es válido")
        return v

    @field_validator('classroom')
    def validate_classroom(cls, v):
        # No validamos contra una lista específica ya que hay 97 aulas
        # pero verificamos que el formato sea razonable
        if not v or len(v) < 1 or len(v) > 4:
            raise ValueError(
                "El identificador del aula debe tener entre 1 y 4 caracteres")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "pretest": 60.0,
                "n_student": 22.0,
                "school": "GOOBU",
                "school_setting": "Urban",
                "school_type": "Public",
                "classroom": "18K",
                "teaching_method": "Standard",
                "gender": "Male",
                "lunch": "Does not qualify",
                "student_id": "S12345"
            }
        }


class BatchStudentScoreData(BaseModel):
    """Esquema para los datos de entrada en lote para predicción de puntajes."""
    students: List[StudentScoreData] = Field(..., min_items=1,
                                             description="Lista de datos de estudiantes")

    class Config:
        json_schema_extra = {
            "example": {
                "students": [
                    {
                        "pretest": 60.0,
                        "n_student": 22.0,
                        "school": "GOOBU",
                        "school_setting": "Urban",
                        "school_type": "Public",
                        "classroom": "18K",
                        "teaching_method": "Standard",
                        "gender": "Male",
                        "lunch": "Does not qualify",
                        "student_id": "S12345"
                    },
                    {
                        "pretest": 63.0,
                        "n_student": 21.0,
                        "school": "QOQTS",
                        "school_setting": "Suburban",
                        "school_type": "Public",
                        "classroom": "A93",
                        "teaching_method": "Experimental",
                        "gender": "Female",
                        "lunch": "Qualifies for reduced/free lunch",
                        "student_id": "S12346"
                    }
                ]
            }
        }


class ScoreFeatureImportance(BaseModel):
    """Esquema para la importancia de las características en la predicción de puntajes."""
    feature: str
    coefficient: float
    importance: float


class ScorePredictionResponse(BaseModel):
    """Esquema para la respuesta de predicción individual de puntaje."""
    prediction: float = Field(
        ..., description="Puntaje predicho para el posttest", ge=32.0, le=97.0)
    processing_time_ms: int
    top_features: Optional[List[ScoreFeatureImportance]] = None


class ScoreCategory(str, Enum):
    """Categorías para los puntajes predichos."""
    EXCELENTE = "Excelente"
    BUENO = "Bueno"
    SATISFACTORIO = "Satisfactorio"
    NECESITA_MEJORA = "Necesita mejora"
    INSUFICIENTE = "Insuficiente"


class ScoreExplanationData(BaseModel):
    """Esquema para los datos de explicación de una predicción de puntaje."""
    summary: str
    percentile: str
    key_factors: List[str]


class ScorePredictionWithExplanation(BaseModel):
    """Esquema para la respuesta de predicción de puntaje con explicación."""
    prediction: float = Field(
        ..., description="Puntaje predicho para el posttest", ge=32.0, le=97.0)
    category: ScoreCategory
    percentile: int = Field(..., ge=1, le=100)
    explanation: ScoreExplanationData
    top_features: List[ScoreFeatureImportance]
    processing_time_ms: int


class ScoreRecommendationItem(BaseModel):
    """Esquema para un ítem de recomendación basado en el puntaje predicho."""
    area: str
    recommendation: str


class ScorePredictionWithRecommendations(BaseModel):
    """Esquema para la respuesta de predicción de puntaje con recomendaciones."""
    student_id: Optional[str] = None
    prediction: float = Field(
        ..., description="Puntaje predicho para el posttest", ge=32.0, le=97.0)
    category: ScoreCategory
    percentile: int = Field(..., ge=1, le=100)
    recommendations: List[ScoreRecommendationItem]
    student_data: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None


class ScoreCategoryDistribution(BaseModel):
    """Esquema para la distribución de categorías en el resultado del lote."""
    category: ScoreCategory
    count: int
    percentage: float


class BatchScorePredictionResult(BaseModel):
    """Esquema para un resultado individual de puntaje dentro de un lote."""
    student_index: int
    student_id: Optional[str] = None
    prediction: float = Field(
        ..., description="Puntaje predicho para el posttest", ge=32.0, le=97.0)
    category: Optional[ScoreCategory] = None
    percentile: Optional[int] = Field(None, ge=1, le=100)
    student_data: Optional[Dict[str, Any]] = None


class BatchScorePredictionStats(BaseModel):
    """Esquema para las estadísticas de predicción de puntajes en lote."""
    count: int
    average_prediction: Optional[float] = None
    min_prediction: Optional[float] = None
    max_prediction: Optional[float] = None
    category_distribution: Optional[List[ScoreCategoryDistribution]] = None
    processing_time_ms: int


class BatchScorePredictionResponse(BaseModel):
    """Esquema para la respuesta de predicción de puntajes en lote."""
    results: List[BatchScorePredictionResult]
    stats: BatchScorePredictionStats
