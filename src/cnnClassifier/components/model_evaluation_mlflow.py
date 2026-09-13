import os
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.metrics import precision_score, recall_score, f1_score
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
            validation_split=0.20  # Đã đồng bộ 0.20 với model_training.py
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        
        # 1. Tính Loss & Accuracy cơ bản
        self.score = self.model.evaluate(self.valid_generator)

        self.valid_generator.reset()
        
        # 2. Chạy predict để lấy nhãn dự đoán và tính Precision, Recall, F1
        predictions = self.model.predict(self.valid_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.valid_generator.classes

        self.precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        self.recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        self.f1_score = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

        self.save_score()

    def save_score(self):
        scores = {
            "loss": float(self.score[0]), 
            "accuracy": float(self.score[1]),
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score
        }
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        os.environ["MLFLOW_TRACKING_URI"] = self.config.mlflow_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = "Salanky79"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "e6fee255f15d712adf3a3d40f9ae781f114f1dd7"

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)

        mlflow.set_experiment("Kidney_Disease_Classification")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name="ResNet50"):
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": float(self.score[0]), 
                "accuracy": float(self.score[1]),
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score
            })

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="ResNet50Model")
            else:
                mlflow.keras.log_model(self.model, "model")