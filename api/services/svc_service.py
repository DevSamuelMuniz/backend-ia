import os
import joblib
from api.models.predict import SVCInput

class SVCService:
    def __init__(self):
        model_path = os.path.join("api", "model", "svc_model.pkl")
        self.model = joblib.load(model_path)

    def predict(self, payload: SVCInput):
        prediction = self.model.predict([payload.features])[0]
        return {
            "prediction": int(prediction),
        }
