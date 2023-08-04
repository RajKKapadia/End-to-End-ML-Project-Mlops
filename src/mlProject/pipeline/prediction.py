import joblib
from pathlib import Path

import numpy as np

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    def predict(self, data:np.array):
        prediction = self.model.predict(data)

        return prediction
