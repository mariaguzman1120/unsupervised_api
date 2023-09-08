import uvicorn
import numpy as np
import os
import joblib

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO

def read_images(file):
    image = Image.open(BytesIO(file))
    return image

def predict_image(image):
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.array(image)
    image = image.reshape(1, -1)
    image = image.astype(float) / 255.0

    path = os.path.join('input', 'model.pkl')
    model = joblib.load(path)

    prediction = model.predict(image)

    return str(prediction[0])

app = FastAPI()

@app.post('/predict', status_code=200)
async def predict(file:UploadFile=File(...)):
    image = read_images(await file.read())
    predict = predict_image(image)
    return predict
