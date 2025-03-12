# Tabla de contenidos

1. [Nombre](#Nombre)
2. [Descripción](#descripción)
3. [Arquitectura del proyecto](#arquitectura-del-proyecto)
4. [Proceso de Desarrollo](#proceso-de-desarrollo)
5. [Funcionalidades](#Funcionalidades)
6. [Estado del proyecto](#Estado-del-proyecto)
7. [Repositorios](#repositorios)
8. [Agradecimientos](#Agradecimientos)

## 1.Detección de patrones de bienestar y rendimiento estudiantil

## 2.Descripcion

El Identificador de Patrones de Rendimiento y Bienestar Estudiantil es una innovadora plataforma de análisis impulsada por tecnologías de Machine Learning, diseñada para apoyar a las instituciones educativas en la identificación y comprensión de los factores clave que impactan el rendimiento académico y el bienestar psicológico de los estudiantes.

### Objetivos del Proyecto

* Identificar los parámetros clave que influyen en el rendimiento académico y el bienestar estudiantil.

* Analizar las relaciones e impactos entre diferentes factores para generar predicciones precisas.

* Diseñar un sistema que sea escalable y personalizable, adaptable a diversos contextos educativos.

* Proporcionar una API que facilite la integración con sistemas de gestión educativa existentes.

* Generar recomendaciones específicas basadas en datos para mejorar el rendimiento académico y el bienestar psicológico de los estudiantes.

### Conjuntos de Datos Utilizados

A- Rendimiento Estudiantil: Factores relacionados con hábitos de estudio, asistencia, uso de recursos educativos y otros aspectos que impactan el desempeño académico general.

B- Score Académico: Datos enfocados en calificaciones, resultados de evaluaciones y métricas específicas de desempeño académico.

C- Indicadores de Depresión: Información sobre métricas relacionadas con el bienestar psicológico y emocional de los estudiantes, clave para evaluar su salud mental.

Estos conjuntos de datos se encuentran en el directorio `/app/db/` de la aplicación:
- `StudentPerformanceFactors.csv` - [https://www.kaggle.com/datasets/lainguyn123/student-performance-factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- `ExamScore.csv` [https://www.kaggle.com/datasets/kwadwoofosu/predict-test-scores-of-students](https://www.kaggle.com/datasets/kwadwoofosu/predict-test-scores-of-students)
- `StudentDepression.csv` [https://www.kaggle.com/datasets/hopesb/student-depression-dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)

  

## 3.Arquitectura del proyecto

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

## 4.Proceso de Desarrollo

### A- Elección y Preprocesamiento de Datos

- Identificación de conjuntos de datos relevantes para el rendimiento académico y bienestar estudiantil
- Limpieza y normalización de datos (manejo de valores nulos, outliers, etc.)
- Análisis exploratorio para identificar tendencias y correlaciones iniciales
- Preparación de conjuntos de entrenamiento y validación

### B- Elección de Modelos de Machine Learning

- Evaluación de diferentes algoritmos supervisados para predicción (Regresión lineal, Random Forest)
- Optimización de hiperparámetros para cada modelo
- Métricas utilizadas para evaluar los modelos Precisión, Recall, F1-Score (clasificación); RMSE, R² (regresión).

### C- Identificación de Casos de Uso

- Predicción de Rendimiento Académico Basado en Factores Comportamentales El sistema utiliza modelos de Machine Learning para analizar comportamientos estudiantiles (como hábitos de estudio, participación en clases y manejo del tiempo) y predecir el rendimiento académico con alta precisión.
- Predicción de Probabilidad de Bajo Rendimiento en Exámenes Incorporación de algoritmos predictivos capaces de identificar estudiantes con alta probabilidad de obtener resultados bajos en exámenes, permitiendo una intervención temprana y oportuna.
- Generación de Recomendaciones Personalizadas para Mejora Basándose en los análisis y predicciones realizadas, el sistema genera recomendaciones específicas y personalizadas para cada estudiante, enfocándose en áreas de mejora tanto académica como de bienestar.

### D- Desarrollo de API

- Utilizacion del Framework: FastAPI
- Implementación de endpoints RESTful para cada modelo
- Documentación de API con especificaciones OpenAPI/Swagger
- Validación de entradas y manejo de errores

### E- Implementación en Módulo de Prueba

- Desarrollo de interfaz de usuario en Laravel para demostración
- Integración con la API de predicción
- Implementación de flujos de asesoramiento estudiantil
- Pruebas de usabilidad y experiencia de usuario
- Documentación de proceso de integración

### F- Endpoints API

La API ofrece varios endpoints de predicción, cada uno especializado en un aspecto diferente:

1. **/prediction**: Genera predicciones individuales basadas en datos de estudiantes
2. **/prediction/batch**: Procesa múltiples predicciones en una sola solicitud
3. **/prediction/explain**: Proporciona predicciones con explicaciones sobre los factores más influyentes
4. **/prediction/recommendation**: Ofrece predicciones con recomendaciones personalizadas

Cada conjunto de datos (rendimiento, score, depresión) tiene su propio modelo y endpoints específicos.

### G- Caso de uso principal: Asesoramiento Estudiantil

El sistema se integra en departamentos de asesoramiento estudiantil para:

1. Identificar tempranamente estudiantes en riesgo académico
2. Proporcionar recomendaciones personalizadas a consejeros y tutores
3. Monitorear la efectividad de intervenciones a lo largo del tiempo
4. Apoyar decisiones basadas en datos para mejorar políticas educativas

### H- Demostración: Módulo de Asesoramiento

Se ha desarrollado un prototipo funcional utilizando Laravel que demuestra cómo integrar la API en un sistema existente. Este módulo muestra las principales funcionalidades:

- Visualización de predicciones para asesores
- Interfaz para ingreso de datos estudiantiles
- Generación de recomendaciones personalizadas

## 5.Estado del proyecto

El proyecto Identificador de Patrones de Rendimiento y Bienestar Estudiantil ha avanzado significativamente en su desarrollo ya que el sistema se integra en departamentos de asesoramiento estudiantil para:

- A. Identificar tempranamente estudiantes en riesgo académico
- B. Proporcionar recomendaciones personalizadas a consejeros y tutores
- C. Monitorear la efectividad de intervenciones a lo largo del tiempo
- D. Apoyar decisiones basadas en datos para mejorar políticas educativas

A continuación, se detalla el estado de cada componente: 

- API Funcional: La API que soporta la plataforma está completamente desarrollada y operativa. Permite integrar sus propias bases de datos y obtener insights en tiempo real. La API cuenta con endpoints bien documentados para facilitar su uso y adaptación a diferentes entornos tecnológicos.
  
- Se ha desarrollado un prototipo funcional utilizando Laravel que demuestra cómo integrar la API en un sistema existente. Este módulo muestra las principales funcionalidades:

- Visualización de predicciones para asesores
- Interfaz para ingreso de datos estudiantiles
- Generación de recomendaciones personalizadas
- Documentación detallada de los endpoints y ejemplos de uso.

- Chatbot de Asesoramiento (En Proceso):
Actualmente, se encuentra en desarrollo el chatbot de asesoramiento, el cual estará integrado en la página web. Este chatbot tendrá como objetivo guiar a los usuarios en el uso de la plataforma, responder preguntas frecuentes y proporcionar recomendaciones personalizadas basadas en los datos analizados. Aunque aún está en fase de implementación, se espera que esté operativo en las próximos dias.

### Próximos Pasos

- Finalizar el desarrollo e integración del chatbot de asesoramiento.
- Realizar pruebas adicionales con un mayor volumen de datos y usuarios.
- **Personalización avanzada**: Permitir que especialistas definan sus propios parámetros de evaluación
- **Aprendizaje continuo**: Implementación de técnicas de aprendizaje incremental para mejorar modelos con nuevos datos
- **Expansión de conjuntos de datos**: Incorporación de fuentes adicionales como datos socioeconómicos, actividades extracurriculares, etc.
- **Análisis predictivo a largo plazo**: Seguimiento de trayectorias académicas completas
- **Interfaces especializadas**: Desarrollo de dashboards específicos para diferentes roles (estudiantes, profesores, asesores)

## 6.Funcionalidades extras:

A. Integración del proyecto en una página web
   - **Frontend**: Laravel (para la interfaz web y la interacción del usuario).
   - **Backend**: FastAPI (para el chatbot y la API de análisis de datos).
   - **Comunicación**: Laravel se comunicará con FastAPI mediante solicitudes HTTP (RESTful API).

B. Creacion de un Chatbot 
- Scikit-learn
- SpaCy
- Python
- FastAPI
    
## 7.Repositorios

El proyecto está dividido en dos repositorios principales:

1. **API de Rendimiento Académico**
   - [https://github.com/AlexanderAzocar/-Detecting-patterns-of-student-well-being-and-performance/tree/main](https://github.com/AlexanderAzocar/-Detecting-patterns-of-student-well-being-and-performance/tree/main)
   - Implementación completa de la API y modelos finales

2. **Interfaz de Asesoramiento (Caso de Uso)**
   - [https://github.com/d4na3l/student-performance-interface](https://github.com/d4na3l/student-performance-interface)
   - Prototipo funcional que demuestra la integración con sistemas educativos

## 8.Agradecimientos

Agradecemos a la profesora Jenny Remolina y a nuestro tutor Alvaro Arauz por acompañarnos en todo el camino orientandonos en la creacion de este proyecto por otro lado agradecemos a  IBM SPSS por proporcionar los datos que sustentan este proyecto.
