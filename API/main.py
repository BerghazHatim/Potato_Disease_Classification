from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()

#model loading
MODEL = tf.keras.models.load_model("saved_models/my_model.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

#Ping for checking if server Up or down
@app.get("/ping")
async def ping():
    return "hello, this is working"

#Function for reading Image into Numpy array
def read_file_as_image(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

#Port for reading user's input
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = read_file_as_image(await file.read())
    img_bacth = np.expand_dims(image,0)
    inference = MODEL.predict(img_bacth)
    inference_class= CLASS_NAMES[np.argmax(inference[0])]
    confidence = np.max(inference[0])
    #return inference_class, confidence
    return {
        'class' : inference_class,
        'confidence' : float(confidence)
    }

#Launching app on localhost
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

