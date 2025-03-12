from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional


class StudentPerformanceData(BaseModel):
    """Esquema para los datos de entrada de un estudiante."""
    Hours_Studied: float = Field(...,
                                 description="Horas de estudio (horas/semana)", ge=0, le=168)
    Attendance: float = Field(..., description="Asistencia (%)", ge=0, le=100)
    Sleep_Hours: float = Field(..., description="Horas de sueño", ge=0, le=24)
    Previous_Scores: float = Field(...,
                                   description="Puntuaciones previas", ge=0, le=100)
    Tutoring_Sessions: float = Field(...,
                                     description="Número de sesiones de tutoría (por mes)", ge=0)
    Physical_Activity: float = Field(
        ..., description="Actividad física (horas/semana)", ge=0, le=168)

    Access_to_Resources: str = Field(...,
                                     description="Acceso a recursos (Low, Medium, High)")
    Parental_Involvement: str = Field(
        ..., description="Participación de los padres (Low, Medium, High)")
    Motivation_Level: str = Field(...,
                                  description="Nivel de motivación (Low, Medium, High)")
    Family_Income: str = Field(...,
                               description="Ingresos familiares (Low, Medium, High)")

    Peer_Influence: str = Field(
        ..., description="Influencia de compañeros (Negative, Neutral, Positive)")

    Extracurricular_Activities: str = Field(
        ..., description="Actividades extracurriculares (Yes, No)")
    Internet_Access: str = Field(...,
                                 description="Acceso a internet (Yes, No)")
    Learning_Disabilities: str = Field(...,
                                       description="Dificultades de aprendizaje (Yes, No)")

    School_Type: str = Field(...,
                             description="Tipo de escuela (Private, Public)")
    Gender: str = Field(..., description="Género (Male, Female)")

    @field_validator('Access_to_Resources', 'Parental_Involvement', 'Motivation_Level', 'Family_Income')
    def validate_ordinal(cls, v):
        allowed = ["Low", "Medium", "High"]
        if v not in allowed:
            raise ValueError(f"Debe ser uno de: {', '.join(allowed)}")
        return v

    @field_validator('Peer_Influence')
    def validate_peer_influence(cls, v):
        allowed = ["Negative", "Neutral", "Positive"]
        if v not in allowed:
            raise ValueError(f"Debe ser uno de: {', '.join(allowed)}")
        return v

    @field_validator('Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities')
    def validate_binary(cls, v):
        allowed = ["Yes", "No"]
        if v not in allowed:
            raise ValueError(f"Debe ser uno de: {', '.join(allowed)}")
        return v

    @field_validator('Gender')
    def validate_gender(cls, v):
        allowed = ["Male", "Female"]
        if v not in allowed:
            raise ValueError(f"Debe ser uno de: {', '.join(allowed)}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "Hours_Studied": 10.5,
                "Attendance": 95.0,
                "Sleep_Hours": 7.5,
                "Previous_Scores": 82.0,
                "Tutoring_Sessions": 5.0,
                "Physical_Activity": 4.0,
                "Access_to_Resources": "High",
                "Parental_Involvement": "Medium",
                "Motivation_Level": "High",
                "Family_Income": "Medium",
                "Peer_Influence": "Positive",
                "Extracurricular_Activities": "Yes",
                "Internet_Access": "Yes",
                "Learning_Disabilities": "No",
                "School_Type": "Public",
                "Gender": "Female"
            }
        }


class BatchStudentData(BaseModel):
    """Esquema para los datos de entrada en lote."""
    students: List[StudentPerformanceData] = Field(..., min_items=1,
                                                   description="Lista de datos de estudiantes")

    class Config:
        json_schema_extra = {
            "example": {
                "students": [
                    {
                        "Hours_Studied": 10.5,
                        "Attendance": 95.0,
                        "Sleep_Hours": 7.5,
                        "Previous_Scores": 82.0,
                        "Tutoring_Sessions": 5.0,
                        "Physical_Activity": 4.0,
                        "Access_to_Resources": "High",
                        "Parental_Involvement": "Medium",
                        "Motivation_Level": "High",
                        "Family_Income": "Medium",
                        "Peer_Influence": "Positive",
                        "Extracurricular_Activities": "Yes",
                        "Internet_Access": "Yes",
                        "Learning_Disabilities": "No",
                        "School_Type": "Public",
                        "Gender": "Female"
                    },
                    {
                        "Hours_Studied": 8.0,
                        "Attendance": 88.0,
                        "Sleep_Hours": 8.5,
                        "Previous_Scores": 75.0,
                        "Tutoring_Sessions": 3.0,
                        "Physical_Activity": 2.0,
                        "Access_to_Resources": "Medium",
                        "Parental_Involvement": "Low",
                        "Motivation_Level": "Medium",
                        "Family_Income": "Low",
                        "Peer_Influence": "Neutral",
                        "Extracurricular_Activities": "No",
                        "Internet_Access": "Yes",
                        "Learning_Disabilities": "No",
                        "School_Type": "Public",
                        "Gender": "Male"
                    }
                ]
            }
        }


class FeatureImportance(BaseModel):
    """Esquema para la importancia de las características."""
    feature: str
    coefficient: float
    importance: float


class PredictionResponse(BaseModel):
    """Esquema para la respuesta de predicción individual."""
    prediction: float
    processing_time_ms: int
    top_features: Optional[List[FeatureImportance]] = None


class ExplanationData(BaseModel):
    """Esquema para los datos de explicación de una predicción."""
    summary: str
    percentile: str
    key_factors: List[str]


class PredictionWithExplanation(BaseModel):
    """Esquema para la respuesta de predicción con explicación."""
    prediction: float
    category: str
    percentile: int
    explanation: ExplanationData
    top_features: List[FeatureImportance]
    processing_time_ms: int


class RecommendationItem(BaseModel):
    """Esquema para un ítem de recomendación."""
    area: str
    recommendation: str


class PredictionWithRecommendations(BaseModel):
    """Esquema para la respuesta de predicción con recomendaciones."""
    student_id: Optional[str] = None
    prediction: float
    category: str
    percentile: int
    recommendations: List[RecommendationItem]
    student_data: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None


class CategoryDistribution(BaseModel):
    """Esquema para la distribución de categorías en el resultado del lote."""
    category: str
    count: int
    percentage: float


class BatchPredictionResult(BaseModel):
    """Esquema para un resultado individual dentro de un lote."""
    student_index: int
    prediction: float
    category: Optional[str] = None
    percentile: Optional[int] = None
    student_data: Optional[Dict[str, Any]] = None


class BatchPredictionStats(BaseModel):
    """Esquema para las estadísticas de predicción en lote."""
    count: int
    average_prediction: Optional[float] = None
    min_prediction: Optional[float] = None
    max_prediction: Optional[float] = None
    category_distribution: Optional[List[CategoryDistribution]] = None
    processing_time_ms: int


class BatchPredictionResponse(BaseModel):
    """Esquema para la respuesta de predicción en lote."""
    results: List[BatchPredictionResult]
    stats: BatchPredictionStats
