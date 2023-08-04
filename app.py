import numpy as np
from fastapi import FastAPI, Request
import gradio as gr

from mlProject.pipeline.prediction import PredictionPipeline
from templates.gradio_ui import demo

app = FastAPI()

@app.get('/',)
def home():
    return 'UI is running at /gradio', 200

@app.post('/predict')
async def predict(request: Request):
    body = await request.json()
    fixed_acidity = float(body['fixed_acidity'])
    volatile_acidity = float(body['volatile_acidity'])
    citric_acid = float(body['citric_acid'])
    residual_sugar = float(body['residual_sugar'])
    chlorides = float(body['chlorides'])
    free_sulfur_dioxide = float(body['free_sulfur_dioxide'])
    total_sulfur_dioxide = float(body['total_sulfur_dioxide'])
    density = float(body['density'])
    pH = float(body['pH'])
    sulphates = float(body['sulphates'])
    alcohol = float(body['alcohol'])
    
    data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
    data = np.array(data).reshape(1, 11)
    
    obj = PredictionPipeline()
    predict = obj.predict(data)

    return f'The wine quality is {round(float(predict), 2)}.', 200

app = gr.mount_gradio_app(app, demo, '/gradio')
