import os
from pathlib import Path
import time
from zipfile import ZipFile
import urllib.request as request
from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf


class Training:

  def __init__(self, config: TrainingConfig):
    self.config = config

  def get_base_model(self):
    self.model = tf.keras.models.load_model(
        self.config.updated_base_model_path
    )

  def train_valid_generator(self):
    datagenerator_kwargs = dict(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        validation_split=0.20,  # Dành 80% dữ liệu cho train để fine-tune tốt hơn
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
          rotation_range=20,  # Giảm xoay xuống 20 độ tránh làm biến dạng góc cắt CT
          horizontal_flip=True,
          width_shift_range=0.1,
          height_shift_range=0.1,
          shear_range=0.1,
          zoom_range=0.2,
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
    self.steps_per_epoch = (
        self.train_generator.samples // self.train_generator.batch_size
    )
    self.validation_steps = (
        self.valid_generator.samples // self.valid_generator.batch_size
    )

    # 1. Thêm Callback EarlyStopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,  # Ngắt nếu val_loss không giảm sau 3 epochs
        restore_best_weights=True,  # Giữ lại trọng số ở epoch có val_loss thấp nhất
        verbose=1,
    )

    # 2. Huấn luyện kèm Callback
    self.model.fit(
        self.train_generator,
        epochs=self.config.params_epochs,
        steps_per_epoch=self.steps_per_epoch,
        validation_steps=self.validation_steps,
        validation_data=self.valid_generator,
        callbacks=[early_stopping],  # Truyền Callback vào fit()
    )

    # 3. Lưu lại mô hình chuẩn nhất
    self.save_model(
        path=self.config.trained_model_path, model=self.model
    )