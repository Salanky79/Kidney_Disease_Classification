import os
import numpy as np
import tensorflow as tf

# Tạo hàm loss giả để Keras vượt qua bước kiểm tra metadata của file .h5
def dummy_loss(y_true, y_pred):
    return y_pred

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model_path = os.path.join("artifacts", "training", "model.h5")
        
        # Truyền custom_objects chứa 'loss_fn' giả lập
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'loss_fn': dummy_loss}, 
            compile=False
        )

        # Preprocess ảnh
        imagename = self.filename
        test_image = tf.keras.preprocessing.image.load_img(imagename, target_size=(224, 224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = tf.keras.applications.resnet50.preprocess_input(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Dự đoán
        result = np.argmax(model.predict(test_image), axis=1)

        labels = {0: 'Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}
        prediction = labels[result[0]]

        return [{"image": prediction}]