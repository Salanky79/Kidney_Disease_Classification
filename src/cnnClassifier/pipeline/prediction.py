import os
import numpy as np
import tensorflow as tf


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # 1. Trỏ đường dẫn mô hình đã train trong artifacts
        model_path = os.path.join("artifacts", "training", "model.h5")
        model = tf.keras.models.load_model(model_path)

        # 2. Load ảnh và chuẩn hóa [0, 1]
        imagename = self.filename
        test_image = tf.keras.preprocessing.image.load_img(imagename, target_size=(224, 224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = tf.keras.applications.resnet50.preprocess_input(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # 3. Dự đoán và lấy index cao nhất
        result = np.argmax(model.predict(test_image), axis=1)

        # 4. Ánh xạ nhãn theo Alphabet
        labels = {0: 'Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}
        prediction = labels[result[0]]

        return [{"image": prediction}]