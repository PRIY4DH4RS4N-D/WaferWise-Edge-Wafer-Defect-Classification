import os
import numpy as np
import tensorflow as tf
import cv2

# ================= CONFIG =================
MODEL_PATH = "wafer_classifier_float32.tflite"
TEST_DIR = "test_dataset/Hackathon_phase3_prediction_dataset"
IMG_SIZE = (128, 128)

LOG_FILE = "predictions_log.txt"

LABELS = [
    "BRIDGE", "CLEAN_CRACK", "CLEAN_LAYER", "CLEAN_VIA",
    "CMP", "CRACK", "LER", "OPEN", "OTHERS", "PARTICLE", "VIA"
]

# ==========================================

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Model Loaded")


# ================= PREPROCESS =================
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot load {img_path}")

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img


# ================= TEST + LOG =================
image_files = sorted(os.listdir(TEST_DIR))

total = 0

with open(LOG_FILE, "w") as f:

    # 🔥 HEADER
    f.write("=" * 50 + "\n")
    f.write("      WAFER DEFECT PREDICTION LOG\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"{'Image Name':<20} | {'Predicted Class':<15}\n")
    f.write("-" * 50 + "\n")

    for img_name in image_files:
        img_path = os.path.join(TEST_DIR, img_name)

        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        try:
            input_data = preprocess_image(img_path)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output = interpreter.get_tensor(output_details[0]['index'])[0]

            pred_class = np.argmax(output)
            pred_label = LABELS[pred_class]

            # ✅ NICE FORMATTED LINE
            line = f"{img_name:<20} | {pred_label:<15}\n"
            f.write(line)

            print(line.strip())
            total += 1

        except Exception as e:
            print(f"❌ Error with {img_name}: {e}")

    # 🔥 FOOTER
    f.write("\n" + "=" * 50 + "\n")
    f.write(f"Total Images: {total}\n")
    f.write("=" * 50 + "\n")

print(f"\n✅ Clean log saved as: {LOG_FILE}")
