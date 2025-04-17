import os
import joblib
from api.models.predict import SVCInput

class SVCService:
    def __init__(self):
        model_path = os.path.join("api", "model", "svc_model.pkl")
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]

    def predict(self, payload: SVCInput):
        prediction = self.model.predict([payload.features])[0]
        return {
            "prediction": int(prediction),
        }
