from pathlib import Path
import os
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

# =====================================================================
# Tự định nghĩa Alpha-Balanced Focal Loss đồng bộ toàn hệ thống
# =====================================================================
def categorical_focal_loss(alpha=[1.0, 1.0, 2.5, 2.0], gamma=2.0):
    alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Thêm keepdims=True để giữ shape (batch_size, 1), khớp hoàn toàn với y_pred (batch_size, 4)
        alpha_factor = tf.reduce_sum(y_true * alpha_tensor, axis=-1, keepdims=True)
        
        focal_weight = alpha_factor * tf.pow(1.0 - y_pred, gamma)
        cross_entropy = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(focal_weight * cross_entropy, axis=-1)

    return loss_fn


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

        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.Dropout(0.5)(x)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
        )(x)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        # Compile với Alpha-Balanced Focal Loss đồng bộ
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=categorical_focal_loss(alpha=[1.0, 1.0, 2.5, 2.0], gamma=2.0),
            metrics=["accuracy"],
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=30,  # Unfreeze 30 lớp cuối (mở trọn vẹn conv5_block)
            learning_rate=self.config.params_learning_rate,
        )

        self.save_model(
            path=self.config.updated_base_model_path, model=self.full_model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.models.Model):
        model.save(path)