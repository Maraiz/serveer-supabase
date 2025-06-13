import sys
import json
import numpy as np
from PIL import Image
import os

import tensorflow as tf

# Load model .tflite
model_path = os.path.join(os.path.dirname(__file__), 'model.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Ambil info input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class names
class_names_path = os.path.join(os.path.dirname(__file__), 'class_names.json')
try:
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    if isinstance(class_names, list):
        class_mapping = {str(i): name for i, name in enumerate(class_names)}
    else:
        class_mapping = class_names
except:
    class_mapping = {str(i): f"class_{i}" for i in range(22)}

if __name__ == "__main__":
    try:
        if len(sys.argv) > 2 and sys.argv[1] == "image":
            image_path = sys.argv[2]

            # Validasi file exists
            if not os.path.exists(image_path):
                result = {"status": "error", "error": f"Image file not found: {image_path}"}
                print(json.dumps(result))
                sys.stdout.flush()  # TAMBAHKAN INI
                sys.exit(1)

            # Info ukuran input model
            height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
            channels = input_details[0]['shape'][3]

            # Load dan preprocessing gambar
            image = Image.open(image_path)
            if channels == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
            image = image.resize((width, height))
            input_data = np.array(image).reshape(1, height, width, channels).astype(np.float32) / 255.0

            # Set input ke interpreter
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            predicted_class_idx = int(np.argmax(output_data))
            predicted_prob = float(output_data[predicted_class_idx])
            predicted_label = class_mapping.get(str(predicted_class_idx), f"class_{predicted_class_idx}")

            top_3_indices = np.argsort(output_data)[-3:][::-1]
            top_3_predictions = [{
                "class": class_mapping.get(str(idx), f"class_{idx}"),
                "confidence": float(output_data[idx])
            } for idx in top_3_indices]

            result = {
                "predicted_class": predicted_label,
                "confidence": predicted_prob,
                "top_3_predictions": top_3_predictions,
                "status": "success",
                "model_type": "tflite"
            }
            print(json.dumps(result))
            sys.stdout.flush()  # TAMBAHKAN INI
            
        else:
            result = {"status": "error", "error": "Invalid arguments"}
            print(json.dumps(result))
            sys.stdout.flush()  # TAMBAHKAN INI
            
    except Exception as e:
        result = {"status": "error", "error": str(e)}
        print(json.dumps(result))
        sys.stdout.flush()  # TAMBAHKAN INI
        sys.exit(1)
    
    # TAMBAHKAN DELAY KECIL SEBELUM EXIT
    import time
    time.sleep(0.1)