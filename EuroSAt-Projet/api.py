from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

# autoriser les requêtes depuis l'interface web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# classes EuroSAT
classes = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# charger le modèle ONNX
session = ort.InferenceSession('cnn_eurosat.onnx')

@app.get('/')
def home():
    return {'message': 'API CNN EuroSAT'}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # lire l'image
    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    img = img.resize((64, 64))

    # preprocessing — même que dans le notebook
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    img_array = img_array.transpose(2, 0, 1)  # (64,64,3) → (3,64,64)
    img_array = np.expand_dims(img_array, axis=0)  # → (1,3,64,64)

    # prédiction
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})[0]
    pred_idx = int(np.argmax(output))

    return {
        'classe': classes[pred_idx],
        'index': pred_idx,
    }