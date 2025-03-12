import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from app.core.config import settings


class ExamScorePredictModel:
    """
    Modelo de Machine Learning para predecir el puntaje de exámenes (posttest)
    basado en características académicas y demográficas de los estudiantes.
    """

    def __init__(self):
        """
        Inicializa la clase sin cargar ni entrenar el modelo.
        """
        self.model = None
        self.is_trained = False
        self.categorical_columns = ["school", "school_setting", "school_type",
                                    "classroom", "teaching_method", "gender", "lunch"]
        # Para guardar las columnas del modelo después del one-hot encoding
        self.model_columns = None

    @classmethod
    def load_from_file(cls, model_path: Optional[str] = None) -> 'ExamScorePredictModel':
        """
        Carga un modelo previamente entrenado desde un archivo.

        Args:
            model_path: Ruta al archivo del modelo. Si es None, usa settings.EXAM_MODEL_PATH.

        Returns:
            Una instancia de ExamScorePredictModel con el modelo cargado.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            Exception: Si hay un error al cargar el modelo.
        """
        model_path = model_path or settings.EXAM_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo en {model_path}.")

        try:
            instance = cls()
            model_data = joblib.load(model_path)

            # Cargar tanto el modelo como las columnas esperadas
            if isinstance(model_data, tuple) and len(model_data) == 2:
                instance.model, instance.model_columns = model_data
            else:
                # Si solo se guardó el modelo sin las columnas
                instance.model = model_data

            instance.is_trained = True
            return instance
        except Exception as e:
            raise Exception(
                f"Error al cargar el modelo desde {model_path}: {str(e)}")

    @classmethod
    def train_from_dataframe(cls, df: pd.DataFrame) -> Tuple['ExamScorePredictModel', Dict[str, Any]]:
        """
        Entrena un nuevo modelo usando un DataFrame.

        Args:
            df: DataFrame que debe incluir la columna 'posttest' como objetivo.

        Returns:
            Una tupla con una instancia de ExamScorePredictModel con el modelo entrenado
            y un diccionario con métricas de evaluación.

        Raises:
            ValueError: Si el DataFrame no tiene la columna 'posttest' o está vacío.
            Exception: Si hay un error durante el entrenamiento.
        """
        if df.empty:
            raise ValueError("El DataFrame está vacío.")

        if "posttest" not in df.columns:
            raise ValueError(
                "El DataFrame debe contener la columna 'posttest'")

        try:
            instance = cls()

            # Limpiar y preparar los datos
            df_clean = df.drop_duplicates()

            # Eliminar student_id si existe
            if "student_id" in df_clean.columns:
                df_clean = df_clean.drop("student_id", axis=1)

            # Aplicar one-hot encoding a las columnas categóricas
            categorical_cols = [
                col for col in instance.categorical_columns if col in df_clean.columns]
            df_encoded = pd.get_dummies(
                df_clean, columns=categorical_cols, drop_first=True)

            # Separar características y objetivo
            X = df_encoded.drop("posttest", axis=1)
            y = df_encoded["posttest"]

            # Guardar las columnas resultantes del encoding para usarlas en predicciones
            instance.model_columns = X.columns.tolist()

            # Dividir en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, random_state=11)

            # Crear y entrenar el modelo
            instance.model = LinearRegression()
            instance.model.fit(X_train, y_train)
            instance.is_trained = True

            # Evaluar el modelo
            y_pred_train = instance.model.predict(X_train)
            y_pred_test = instance.model.predict(X_test)

            train_score = instance.model.score(X_train, y_train)
            test_score = instance.model.score(X_test, y_test)

            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            evaluation = {
                "train_score": train_score,
                "test_score": test_score,
                "mse_train": mse_train,
                "mse_test": mse_test,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "test_size": len(X_test),
                "train_size": len(X_train)
            }

            return instance, evaluation

        except Exception as e:
            raise Exception(
                f"Error durante el entrenamiento del modelo: {str(e)}")

    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Guarda el modelo entrenado en un archivo.

        Args:
            model_path: Ruta donde guardar el modelo. Si es None, usa settings.EXAM_MODEL_PATH.

        Returns:
            La ruta donde se guardó el modelo.

        Raises:
            ValueError: Si el modelo no ha sido entrenado.
            Exception: Si hay un error al guardar el modelo.
        """
        if not self.is_trained or self.model is None:
            raise ValueError(
                "No se puede guardar un modelo que no ha sido entrenado.")

        model_path = model_path or settings.EXAM_MODEL_PATH

        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        try:
            # Guardar tanto el modelo como las columnas esperadas
            joblib.dump((self.model, self.model_columns), model_path)
            return model_path
        except Exception as e:
            raise Exception(
                f"Error al guardar el modelo en {model_path}: {str(e)}")

    def predict(self, data):
        """
        Predice el puntaje del examen (posttest) para uno o varios estudiantes.

        Args:
            data: Diccionario o DataFrame con las características del estudiante.

        Returns:
            Predicción del puntaje de examen o array de predicciones.

        Raises:
            ValueError: Si el modelo no ha sido entrenado o si los datos de entrada no son válidos.
        """
        if not self.is_trained or self.model is None:
            raise ValueError(
                "El modelo debe ser entrenado o cargado antes de realizar predicciones.")

        # Procesar los datos de entrada
        try:
            if isinstance(data, dict):
                # Si es un diccionario, conviértelo a DataFrame de una sola fila
                processed_data = self._preprocess_data_dict(data)
                prediction = self.model.predict(processed_data)
                return prediction[0]

            elif isinstance(data, pd.DataFrame):
                # Si ya es un DataFrame, preprocésalo
                processed_data = self._preprocess_data_df(data)
                predictions = self.model.predict(processed_data)
                return predictions

            else:
                raise ValueError(
                    "Los datos deben ser un diccionario o un DataFrame.")

        except Exception as e:
            raise ValueError(
                f"Error al procesar los datos para predicción: {str(e)}")

    def _preprocess_data_dict(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocesa un diccionario de datos para adecuarlo al formato esperado por el modelo.

        Args:
            data: Diccionario con las características del estudiante.

        Returns:
            DataFrame preparado para la predicción.
        """
        # Primero, estandarizar y validar los datos de entrada
        data = self.standardize_input(data)
        df_row = pd.DataFrame([data])

        # Eliminar student_id si existe
        if "student_id" in df_row.columns:
            df_row = df_row.drop("student_id", axis=1)

        # Aplicar el mismo preprocesamiento que durante el entrenamiento
        return self._preprocess_data_df(df_row)

    def _preprocess_data_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa un DataFrame para adecuarlo al formato esperado por el modelo.

        Args:
            df: DataFrame con las características del estudiante.

        Returns:
            DataFrame preparado para la predicción.
        """
        if self.model_columns is None:
            raise ValueError(
                "El modelo debe ser entrenado o cargado correctamente con las columnas esperadas.")

        # Aplicar one-hot encoding a las columnas categóricas
        categorical_cols = [
            col for col in self.categorical_columns if col in df.columns]
        df_encoded = pd.get_dummies(
            df, columns=categorical_cols, drop_first=True)

        # Asegurarse de que todas las columnas esperadas por el modelo estén presentes
        for col in self.model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # Añadir columnas faltantes con valor 0

        # Mantener solo las columnas que el modelo espera y en el orden correcto
        result = df_encoded[self.model_columns]

        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extrae la importancia de las características del modelo.

        Returns:
            DataFrame con las características y su importancia.

        Raises:
            ValueError: Si el modelo no ha sido entrenado.
        """
        if not self.is_trained or self.model is None:
            raise ValueError(
                "El modelo debe ser entrenado o cargado para obtener la importancia de características.")

        if self.model_columns is None:
            raise ValueError(
                "No se pueden determinar las características del modelo.")

        coefs = self.model.coef_

        importance_df = pd.DataFrame({
            'feature': self.model_columns,
            'coefficient': coefs,
            'importance': np.abs(coefs)
        }).sort_values('importance', ascending=False)

        return importance_df

    def standardize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estandariza y valida los valores de entrada, manejando diferentes formatos y valores.

        Args:
            data: Diccionario con los datos de entrada.

        Returns:
            Diccionario con valores estandarizados.

        Raises:
            ValueError: Si algún campo numérico requerido no está presente o no es convertible a float.
        """
        result = data.copy()

        # Validar que los campos numéricos requeridos estén presentes y sean convertibles a float
        numeric_fields = ["pretest"]
        for field in numeric_fields:
            if field not in result:
                raise ValueError(f"El campo '{field}' es requerido.")
            try:
                result[field] = float(result[field])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"El campo '{field}' debe ser un valor numérico. Error: {str(e)}")

        # Mapas para estandarización de valores categóricos
        school_map = {
            "a": "A", "b": "B", "c": "C",
            "escuela a": "A", "escuela b": "B", "escuela c": "C"
        }

        setting_map = {
            "urbano": "Urban", "urban": "Urban",
            "rural": "Rural",
            "suburbano": "Suburban", "suburban": "Suburban"
        }

        type_map = {
            "publica": "Public", "pública": "Public", "public": "Public",
            "privada": "Private", "private": "Private",
            "charter": "Charter"
        }

        gender_map = {
            "masculino": "Male", "male": "Male", "m": "Male",
            "femenino": "Female", "female": "Female", "f": "Female"
        }

        lunch_map = {
            "free": "Free", "gratis": "Free",
            "reduced": "Reduced", "reducido": "Reduced",
            "paid": "Paid", "pagado": "Paid"
        }

        # Estandarizar cada campo si está presente
        for field, mapping in [
            ("school", school_map),
            ("school_setting", setting_map),
            ("school_type", type_map),
            ("gender", gender_map),
            ("lunch", lunch_map)
        ]:
            if field in result and isinstance(result[field], str):
                result[field] = mapping.get(
                    result[field].lower(), result[field])

        return result
