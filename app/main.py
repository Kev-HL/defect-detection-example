from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ai_edge_litert import Interpreter
from PIL import Image
import numpy as np
import io

# Load TFLite model
MODEL_PATH = "./best_model.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class mapping
CLASS_MAP = {
    0: "good",
    1: "crack",
    2: "faulty_imprint",
    3: "poke",
    4: "scratch",
    5: "squeeze"
}

app = FastAPI()

def preprocess_image(image_bytes):
    # Decode PNG image using PIL
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((300, 300))
    img_array = np.array(image, dtype=np.float32)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type != "image/png":
        return JSONResponse(status_code=400, content={"error": "Only PNG images are supported."})

    image_bytes = await file.read()
    img = preprocess_image(image_bytes)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Get predictions
    pred_box = interpreter.get_tensor(output_details[0]['index'])
    pred_label_probs = interpreter.get_tensor(output_details[1]['index'])

    pred_label = np.argmax(pred_label_probs, axis=1)
    class_id = int(pred_label[0])
    class_name = CLASS_MAP.get(class_id, "unknown")
    bbox = [float(x) for x in pred_box[0]]

    return {
        "class_id": class_id,
        "class_name": class_name,
        "bbox": bbox
    }