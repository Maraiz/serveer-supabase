import os
import sys
import json
import numpy as np
from tensorflow import keras
from PIL import Image
import io

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'my_model.keras')
model = keras.models.load_model(model_path)

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

def predict_image_from_stdin():
    image_bytes = sys.stdin.buffer.read()
    image = Image.open(io.BytesIO(image_bytes))

    expected_shape = model.input_shape
    if len(expected_shape) != 4:
        raise ValueError("Model is not an image model")

    height, width, channels = expected_shape[1], expected_shape[2], expected_shape[3]

    if channels == 1:
        image = image.convert('L')
    else:
        image = image.convert('RGB')

    image = image.resize((width, height))
    input_data = np.array(image).reshape(1, height, width, channels).astype(np.float32)
    input_data = input_data / 255.0
    return input_data

def predict_tabular_from_args(args):
    features = json.loads(args)
    expected_shape = model.input_shape

    if len(expected_shape) == 4:
        raise ValueError("Model expects image input, not tabular")

    input_data = np.array(features).reshape(1, -1).astype(np.float32)
    return input_data

def make_prediction(input_data):
    prediction = model.predict(input_data, verbose=0)[0]
    predicted_class_idx = int(np.argmax(prediction))
    predicted_prob = float(prediction[predicted_class_idx])
    predicted_label = class_mapping.get(str(predicted_class_idx), f"class_{predicted_class_idx}")

    top_3_indices = np.argsort(prediction)[-3:][::-1]
    top_3_predictions = [
        {
            "class": class_mapping.get(str(idx), f"class_{idx}"),
            "confidence": float(prediction[idx])
        }
        for idx in top_3_indices
    ]

    return {
        "predicted_class": predicted_label,
        "confidence": predicted_prob,
        "top_3_predictions": top_3_predictions,
        "status": "success",
        "model_type": "image" if len(model.input_shape) == 4 else "tabular"
    }

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "image":
            input_data = predict_image_from_stdin()
        else:
            input_data = predict_tabular_from_args(sys.argv[1])

        result = make_prediction(input_data)
        print(json.dumps(result))
    except Exception as e:
        error_result = {
            "error": str(e),
            "status": "error"
        }
        print(json.dumps(error_result))
        sys.exit(1)
