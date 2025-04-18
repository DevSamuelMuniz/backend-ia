import os
import joblib
import numpy as np
import shap
from api.models.predict import SVCInput

class SVCService:
    def __init__(self):
        model_path = os.path.join("api", "model", "svc_model.pkl")
        scaler_path = os.path.join("api", "model", "scaler.pkl")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Base para o SHAP (idealmente: dados reais de treino ou parte deles)
        background = np.zeros((1, self.model.n_features_in_))
        self.explainer = shap.KernelExplainer(self.model.predict_proba, background)

    def predict(self, payload: SVCInput):
        features = np.array(payload.features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]

        # Explicação SHAP
        shap_vals = self.explainer.shap_values(features_scaled)
        shap_values = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
        shap_values = shap_values.tolist()[0]

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "shap_values": shap_values
        }
