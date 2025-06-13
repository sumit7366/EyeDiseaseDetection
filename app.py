import os
import logging
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError, Image

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Load Model
MODEL_PATH = "models/eye_disease_model.h5"
logging.info(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
logging.info("Model loaded successfully!")

CATEGORIES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

# Upload config
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                # Try to load and preprocess the image
                img = image.load_img(filepath, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Make prediction
                prediction = model.predict(img_array)
                result = CATEGORIES[np.argmax(prediction)]

                logging.info(f"Prediction: {result}")
                return render_template("result.html", result=result, img_path=filepath)

            except UnidentifiedImageError:
                logging.error("Uploaded file is not a valid image.")
                return "Uploaded file is not a valid image.", 400
            except Exception as e:
                logging.error(f"Error processing image: {str(e)}")
                return f"Error processing image: {str(e)}", 500

        else:
            return "Unsupported file type. Please upload a PNG or JPG image.", 400

    return render_template("index.html")

if __name__ == "__main__":
    logging.info("Starting Flask app...")
    app.run(debug=True)
