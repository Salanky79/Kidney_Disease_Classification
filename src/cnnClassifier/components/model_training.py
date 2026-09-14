import os
from pathlib import Path
import time
from zipfile import ZipFile
import urllib.request as request
import numpy as np
from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf


# Custom Alpha-Balanced Focal Loss (gamma=2.0, alpha bù đắp dữ liệu yếu)
def categorical_focal_loss(alpha=[1.0, 1.0, 2.5, 2.0], gamma=2.0):
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Thêm keepdims=True để duy trì tensor shape (batch_size, 1), tránh lỗi mismatch shape với y_pred
        alpha_factor = tf.reduce_sum(y_true * alpha_tensor, axis=-1, keepdims=True)
        
        # Tính Focal Weight kết hợp Alpha và Gamma
        focal_weight = alpha_factor * tf.pow(1.0 - y_pred, gamma)
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        return tf.reduce_sum(focal_weight * cross_entropy, axis=-1)

    return loss_fn


class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        # Dùng compile=False để tránh lỗi load Custom Object của Keras
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
            validation_split=0.20,
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                horizontal_flip=True,
                vertical_flip=True, # Bổ sung xoay dọc cho Tumor/Cyst
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.2, # Phóng to hạt sỏi Stone
                fill_mode="nearest",
                **datagenerator_kwargs,
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs,
        )

        print(f"\n[INFO] Class Mapping: {self.train_generator.class_indices}\n")

    @staticmethod
    def save_model(path: Path, model: tf.keras.models.Model):
        model.save(path)

    def train(self):
        # 1. Compile lại mô hình với Alpha-Balanced Focal Loss & LR tối ưu (3e-5)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), 
            loss=categorical_focal_loss(alpha=[1.0, 1.0, 2.5, 2.0], gamma=2.0),
            metrics=["accuracy"],
        )

        # 2. Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        )

        # 3. Fit model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=[early_stopping, reduce_lr],
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )