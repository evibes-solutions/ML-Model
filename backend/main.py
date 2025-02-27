from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


age_model = load_model("age_prediction_model.h5")
gender_model = load_model("gender_prediction_model.h5")


def preprocess_image(image: Image.Image):
    image = image.resize((200, 200))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    processed_image = preprocess_image(image)

  
    age_pred = age_model.predict(processed_image)[0][0]
    gender_pred = gender_model.predict(processed_image)[0][0]
    
    gender = "Male" if gender_pred > 0.5 else "Female"

    return {"age": round(age_pred), "gender": gender}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
