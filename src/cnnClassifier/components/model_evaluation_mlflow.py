import os
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
            validation_split=0.20
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
        # BỔ SUNG compile=False ĐỂ BỎ QUA CUSTOM LOSS KHÔNG TỒN TẠI TRONG KO-GIỜ-SỢ-LỖI
        return tf.keras.models.load_model(path, compile=False)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        
        # COMPILE LAI MÔ HÌNH VỚI LOSS TIÊU CHUẨN ĐỂ CHẠY HÀM evaluate()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        self._valid_generator()
        
        # 1. Tính Loss & Accuracy cơ bản
        self.score = self.model.evaluate(self.valid_generator)

        self.valid_generator.reset()
        
        # 2. Chạy predict để lấy nhãn dự đoán
        predictions = self.model.predict(self.valid_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.valid_generator.classes

        # 3. Tính các chỉ số tổng quan
        self.precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        self.recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        self.f1_score = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

        # 4. In Classification Report chi tiết từng lớp ra Terminal
        target_names = list(self.valid_generator.class_indices.keys())
        self.report_str = classification_report(
            y_true, y_pred, target_names=target_names, digits=4, zero_division=0
        )
        
        print("\n" + "="*20 + " CLASSIFICATION REPORT " + "="*20)
        print(self.report_str)
        print("="*63 + "\n")

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

            # Đẩy file classification_report.txt lên MLflow Artifacts
            mlflow.log_text(self.report_str, artifact_file="classification_report.txt")

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="ResNet50Model")
            else:
                mlflow.keras.log_model(self.model, "model")