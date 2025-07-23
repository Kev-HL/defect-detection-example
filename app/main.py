import numpy as np
import io, os, uuid, time, random, logging, threading
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ai_edge_litert.interpreter import Interpreter
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# Set up logging to show INFO level and above messages
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Mount static files directory for serving images and other assets
# App will raise errors if folders do not exist
# This is handled by the Dockerfile
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and prepare lock for thread-safe inference
MODEL_PATH = "./best_model.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter_lock = threading.Lock()


# Class mapping
CLASS_MAP = {
    0: "good",
    1: "crack",
    2: "faulty_imprint",
    3: "poke",
    4: "scratch",
    5: "squeeze"
}

# Font path for drawing text on images
FONT_PATH = "./fonts/OpenSans-Bold.ttf"

# Max file size for uploads (5 MB)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Function to preprocess the image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((300, 300))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to delete files after a delay
def delete_files_later(files, delay=10):
    time.sleep(delay)
    for f in files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
        except Exception as e:
            logging.error(f"Error deleting file {f}: {e}")

# Function to perform inference on a preprocessed image
def inference(img):
    # Ensure the interpreter is thread-safe
    with interpreter_lock:
        
        # Set the input tensor and invoke the interpreter
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        # Get the prediction results
        pred_box = interpreter.get_tensor(output_details[0]['index'])
        pred_label_probs = interpreter.get_tensor(output_details[1]['index'])

        # Format the prediction results and get the class name
        pred_label = np.argmax(pred_label_probs, axis=1)
        class_id = int(pred_label[0])
        class_name = CLASS_MAP.get(class_id, "unknown")
        bbox = [float(x) for x in pred_box[0]]
        
        return class_id, class_name, bbox

# Function to save an image for later use
def save_image(image_bytes):
    filename = f"{uuid.uuid4()}.png"
    path = f"static/uploads/{filename}"
    with open(path, "wb") as f:
        f.write(image_bytes)
    return filename, path

# Function to draw bounding box and label on the image
def draw_bounding_box(image_path, bbox, class_name):
    box_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(box_img)
    w, h = box_img.size
    xmin, ymin, xmax, ymax = [int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)]
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    try:
        font = ImageFont.truetype(FONT_PATH, 35)
    except:
        font = ImageFont.load_default()
    draw.text((xmin, max(ymin-40, 0)), class_name, fill="red", font=font)

    # Save the visualization image (with bounding box and label)
    filename = f"{uuid.uuid4()}.png"
    path = f"static/results/{filename}"
    box_img.save(path)

    return filename, path

# Root endpoint to render the HTML form
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Render the HTML form with empty image URLs and no result
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "orig_img_url": None,
            "vis_img_url": None,
        }
    )

# Endpoint to handle image prediction (API)
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Check if the uploaded file is a PNG image
        if file.content_type != "image/png":
            return JSONResponse(status_code=400, content={"error": "Only PNG images are supported."})

        # Read the image
        image_bytes = await file.read()

        # Check if the file size exceeds the maximum limit
        if len(image_bytes) > MAX_FILE_SIZE:
            return JSONResponse(status_code=400, content={"error": "File size exceeds the maximum limit of 5 MB."})
        
        # Check if the image is a valid PNG (not just a file with .png extension)
        try:
            img_check = Image.open(io.BytesIO(image_bytes))
            if img_check.format != 'PNG':
                raise ValueError("Not a PNG")
        except (UnidentifiedImageError, ValueError):
            return JSONResponse(status_code=400, content={"error": "Invalid image file."})

        # Preprocess the image
        img = preprocess_image(image_bytes)

        # Run inference on the preprocessed image
        class_id, class_name, bbox = inference(img)

        # Return the prediction results as JSON
        return {
            "class_id": class_id,
            "class_name": class_name,
            "bbox": bbox
        }
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return JSONResponse(status_code=500, content={"error": "Model inference failed."})

# Endpoint to handle image upload and prediction with visualization
@app.post("/upload/", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    try:
        # Check if the uploaded file is a PNG image
        if file.content_type != "image/png":
            result = {"error": "Only PNG images are supported."}
            return templates.TemplateResponse("index.html", {"request": request, "result": result})

        # Read the uploaded image
        image_bytes = await file.read()

        # Check if the file size exceeds the maximum limit
        if len(image_bytes) > MAX_FILE_SIZE:
            return templates.TemplateResponse("index.html", {"request": request, "result": {"error": "File too large (max 5MB)."}})

        # Check if the image is a valid PNG (not just a file with .png extension)
        try:
            img_check = Image.open(io.BytesIO(image_bytes))
            if img_check.format != 'PNG':
                raise ValueError("Not a PNG")
        except (UnidentifiedImageError, ValueError):
            return templates.TemplateResponse("index.html", {"request": request, "result": {"error": "Invalid image file."}})

        # Save the original image
        orig_filename, orig_path = save_image(image_bytes)
    
        # Preprocess the image
        img = preprocess_image(image_bytes)

        # Run inference on the preprocessed image
        class_id, class_name, bbox = inference(img)

        # Draw bounding box
        vis_filename, vis_path = draw_bounding_box(orig_path, bbox, class_name)

        # Prepare the result to be displayed in the HTML template
        result = {
            "class_id": class_id,
            "class_name": class_name,
            "bbox": bbox
        }

        # Schedule deletion of both images after 10 seconds
        if background_tasks is not None:
            background_tasks.add_task(delete_files_later, [orig_path, vis_path], delay=10)

        # Render the HTML template with the result and image URLs
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "orig_img_url": f"/static/uploads/{orig_filename}",
                "vis_img_url": f"/static/results/{vis_filename}",
            }
        )
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "result": {"error": "Model inference failed."}})

# Endpoint to handle random image sampling
@app.post("/random-sample/", response_class=HTMLResponse)
async def random_sample(request: Request, background_tasks: BackgroundTasks = None):
    try:
        # Check if the samples directory exists and contains PNG files
        samples_dir = "static/samples"
        sample_files = [f for f in os.listdir(samples_dir) if f.lower().endswith('.png')]
        if not sample_files:
            result = {"error": "No sample images available."}
            return templates.TemplateResponse("index.html", {"request": request, "result": result})
        
        # Randomly select a sample image and read it
        chosen_file = random.choice(sample_files)
        with open(os.path.join(samples_dir, chosen_file), "rb") as f:
            image_bytes = f.read()
        
        # Save original image
        orig_filename, orig_path = save_image(image_bytes)
        
        # Preprocess the image
        img = preprocess_image(image_bytes)

        # Run inference on the preprocessed image
        class_id, class_name, bbox = inference(img)

        # Draw bounding box
        vis_filename, vis_path = draw_bounding_box(orig_path, bbox, class_name)

        # Prepare the result to be displayed in the HTML template
        result = {
            "class_id": class_id,
            "class_name": class_name,
            "bbox": bbox
        }

        # Schedule deletion of both images after 10 seconds
        if background_tasks is not None:
            background_tasks.add_task(delete_files_later, [orig_path, vis_path], delay=10)

        # Render the HTML template with the result and image URLs
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "orig_img_url": f"/static/uploads/{orig_filename}",
                "vis_img_url": f"/static/results/{vis_filename}",
            }
        )
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "result": {"error": "Model inference failed."}})