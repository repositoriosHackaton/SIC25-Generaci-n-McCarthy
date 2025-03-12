# EduWell, Tu Huella Digital hacia un Aprendizaje Saludable

## Tabla de contenidos

1. [Descripción](#descripción)
2. [Objetivos](#objetivos)
3. [Conjuntos de Datos](#conjuntos-de-datos)
4. [Arquitectura del proyecto](#arquitectura-del-proyecto)
5. [Proceso de Desarrollo](#proceso-de-desarrollo)
6. [Endpoints API](#endpoints-api)
7. [Casos de Uso](#casos-de-uso)
8. [Repositorios](#repositorios)
9. [Instalación y Uso](#instalación-y-uso)

## Descripcion

El Identificador de Patrones de Rendimiento y Bienestar Estudiantil es una innovadora plataforma de análisis impulsada por tecnologías de Machine Learning, diseñada para apoyar a las instituciones educativas en la identificación y comprensión de los factores clave que impactan el rendimiento académico y el bienestar psicológico de los estudiantes.

## Objetivos del Proyecto

* Identificar los parámetros clave que influyen en el rendimiento académico y el bienestar estudiantil.

* Analizar las relaciones e impactos entre diferentes factores para generar predicciones precisas.

* Diseñar un sistema que sea escalable y personalizable, adaptable a diversos contextos educativos.

* Proporcionar una API que facilite la integración con sistemas de gestión educativa existentes.

* Generar recomendaciones específicas basadas en datos para mejorar el rendimiento académico y el bienestar psicológico de los estudiantes.

## Conjuntos de Datos Utilizados

1- Rendimiento Estudiantil: Factores relacionados con hábitos de estudio, asistencia, uso de recursos educativos y otros aspectos que impactan el desempeño académico general.

2- Score Académico: Datos enfocados en calificaciones, resultados de evaluaciones y métricas específicas de desempeño académico.

3- Indicadores de Depresión: Información sobre métricas relacionadas con el bienestar psicológico y emocional de los estudiantes, clave para evaluar su salud mental.

Estos conjuntos de datos se encuentran en el directorio `/app/db/` de la aplicación:
- `StudentPerformanceFactors.csv`
- `ExamScore.csv`
- `StudentDepression.csv`

## Arquitectura del proyecto

```
└── academic-performance-api
    └── app
        └── api
            └── api_v1
                └── api_router.py
                └── endpoints
                    └── performance
                        └── predictions.py
                    └── score
                        └── predictions.py
        └── core
            └── config.py
            └── security.py
        └── db
            └── ExamScore.csv
            └── StudentDepression.csv
            └── StudentPerformanceFactors.csv
        └── models
            └── ExamScorePredictModel.py
            └── StudentPerformanceModel.py
            └── trained
        └── schemas
            └── exam_score_data.py
            └── student_performance_data.py
        └── services
    └── tests
        └── api
        └── models
    └── Notebooks
        └── ped_processor_test
        └── spa_model_test
        └── spa_model_test2
        └── spf_graphics_view
        └── Student Depression Dataset
        └── test_scores
    └── __init__.py
    └── main.py
    └── requirements.txt
```

El proyecto sigue una arquitectura modular y escalable:

- **api**: Contiene los endpoints y rutas de la API
- **core**: Configuraciones generales y seguridad
- **db**: Datos de entrenamiento y validación
- **models**: Modelos de Machine Learning y lógica de predicción
- **schemas**: Definición de estructuras de datos
- **services**: Lógica de negocio y servicios adicionales
- **tests**: Pruebas automatizadas

## Proceso de Desarrollo

### 1. Elección y Preprocesamiento de Datos

- Identificación de conjuntos de datos relevantes para el rendimiento académico y bienestar estudiantil
- Limpieza y normalización de datos (manejo de valores nulos, outliers, etc.)
- Análisis exploratorio para identificar tendencias y correlaciones iniciales
- Preparación de conjuntos de entrenamiento y validación

### 2. Elección de Modelos de Machine Learning

- Evaluación de diferentes algoritmos supervisados para predicción (Regresión lineal, Random Forest)
- Optimización de hiperparámetros para cada modelo
- Métricas utilizadas para evaluar los modelos Precisión, Recall, F1-Score (clasificación); RMSE, R² (regresión).

### 3. Identificación de Casos de Uso

- Predicción de Rendimiento Académico Basado en Factores Comportamentales El sistema utiliza modelos de Machine Learning para analizar comportamientos estudiantiles (como hábitos de estudio, participación en clases y manejo del tiempo) y predecir el rendimiento académico con alta precisión.
- Predicción de Probabilidad de Bajo Rendimiento en Exámenes Incorporación de algoritmos predictivos capaces de identificar estudiantes con alta probabilidad de obtener resultados bajos en exámenes, permitiendo una intervención temprana y oportuna.
- Generación de Recomendaciones Personalizadas para Mejora Basándose en los análisis y predicciones realizadas, el sistema genera recomendaciones específicas y personalizadas para cada estudiante, enfocándose en áreas de mejora tanto académica como de bienestar.

### 4. Desarrollo de API

- Utilizacion del Framework: FastAPI
- Implementación de endpoints RESTful para cada modelo
- Documentación de API con especificaciones OpenAPI/Swagger
- Validación de entradas y manejo de errores

### 5. Implementación en Módulo de Prueba

- Desarrollo de interfaz de usuario en Laravel para demostración
- Integración con la API de predicción
- Implementación de flujos de asesoramiento estudiantil
- Pruebas de usabilidad y experiencia de usuario
- Documentación de proceso de integración

### Endpoints API

La API ofrece varios endpoints de predicción, cada uno especializado en un aspecto diferente:

1. **/prediction**: Genera predicciones individuales basadas en datos de estudiantes
2. **/prediction/batch**: Procesa múltiples predicciones en una sola solicitud
3. **/prediction/explain**: Proporciona predicciones con explicaciones sobre los factores más influyentes
4. **/prediction/recommendation**: Ofrece predicciones con recomendaciones personalizadas

Cada conjunto de datos (rendimiento, score, depresión) tiene su propio modelo y endpoints específicos.

### Caso de uso principal: Asesoramiento Estudiantil

El sistema se integra en departamentos de asesoramiento estudiantil para:

1. Identificar tempranamente estudiantes en riesgo académico
2. Proporcionar recomendaciones personalizadas a consejeros y tutores
3. Monitorear la efectividad de intervenciones a lo largo del tiempo
4. Apoyar decisiones basadas en datos para mejorar políticas educativas

### Demostración: Módulo de Asesoramiento

Se ha desarrollado un prototipo funcional utilizando Laravel que demuestra cómo integrar la API en un sistema existente. Este módulo muestra las principales funcionalidades:

- Visualización de predicciones para asesores
- Interfaz para ingreso de datos estudiantiles
- Generación de recomendaciones personalizadas

## Repositorios

El proyecto está dividido en dos repositorios principales:

1. **API de Rendimiento Académico**
   - [https://github.com/AlexanderAzocar/-Detecting-patterns-of-student-well-being-and-performance/tree/main](https://github.com/AlexanderAzocar/-Detecting-patterns-of-student-well-being-and-performance/tree/main)
   - Implementación completa de la API y modelos finales

2. **Interfaz de Asesoramiento (Caso de Uso)**
   - [https://github.com/d4na3l/student-performance-interface](https://github.com/d4na3l/student-performance-interface)
   - Prototipo funcional que demuestra la integración con sistemas educativos

Para más ejemplos, consulte la documentación de la API disponible en `/docs` después de iniciar el servidor.
