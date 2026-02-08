import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os

# Load model
model = load_model("model/pulmo_model.h5")

LAST_CONV_LAYER = "conv2d_2"

# Dataset root test folder
DATASET_TEST_PATH = "dataset/test"

# Output root folder
OUTPUT_ROOT = "gradcam_results"

# Images per disease
MAX_IMAGES = 5


def generate_gradcam(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (224, 224))

    return heatmap


# Process each disease folder
for disease in os.listdir(DATASET_TEST_PATH):

    disease_folder = os.path.join(DATASET_TEST_PATH, disease)

    if os.path.isdir(disease_folder):

        print(f"\nProcessing Disease: {disease}")

        output_folder = os.path.join(OUTPUT_ROOT, disease)
        os.makedirs(output_folder, exist_ok=True)

        count = 0

        for file in os.listdir(disease_folder):

            if file.lower().endswith((".png", ".jpg", ".jpeg")):

                img_path = os.path.join(disease_folder, file)

                heatmap = generate_gradcam(img_path)

                heatmap_uint8 = np.uint8(255 * heatmap)
                color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                save_path = os.path.join(output_folder, f"gradcam_{file}")
                cv2.imwrite(save_path, color_heatmap)

                print(f"Saved: {save_path}")

                count += 1
                if count >= MAX_IMAGES:
                    break

print("\nGrad-CAM generation completed for ALL diseases.")
