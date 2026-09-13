from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import os
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import tensorflow as tf


class PrepareBaseModel:

  def __init__(self, config: PrepareBaseModelConfig):
    self.config = config

  def get_base_model(self):
    self.model = tf.keras.applications.ResNet50(
        input_shape=self.config.params_image_size,
        weights=self.config.params_weights,
        include_top=self.config.params_include_top,
    )
    self.save_model(path=self.config.base_model_path, model=self.model)

  @staticmethod
  def _prepare_full_model(
      model, classes, freeze_all, freeze_till, learning_rate
  ):
    if freeze_all:
      for layer in model.layers:
        layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
      for layer in model.layers[:-freeze_till]:
        layer.trainable = False

    # 1. Thay Flatten bằng GlobalAveragePooling2D để giảm tham số thừa
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)

    # 2. Thêm Dropout 0.4 ép neuron tự học độc lập
    x = tf.keras.layers.Dropout(0.4)(x)

    # 3. Lớp Dense tích hợp L2 Regularization (l2=0.01)
    # Tác dụng: Phạt các trọng số (weights) có giá trị quá lớn, ngăn mô hình
    # phụ thuộc vào một số đặc trưng cố định để chống Overfitting triệt để.
    prediction = tf.keras.layers.Dense(
        units=classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),  # <-- THÊM L2 REGULARIZATION TẠI ĐÂY
    )(x)

    full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    full_model.summary()
    return full_model

  def update_base_model(self):
    self.full_model = self._prepare_full_model(
        model=self.model,
        classes=self.config.params_classes,
        freeze_all=False,  # Đã đổi thành False
        freeze_till=5,  # Unfreeze 5 layers cuối ResNet50
        learning_rate=self.config.params_learning_rate,
    )

    self.save_model(
        path=self.config.updated_base_model_path, model=self.full_model
    )

  @staticmethod
  def save_model(path: Path, model: tf.keras.models.Model):
    model.save(path)