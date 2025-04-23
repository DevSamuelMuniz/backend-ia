import os
import joblib
import numpy as np
import shap
from predict import SVCInput

BASE_DIR = os.path.dirname(__file__)


class SVCService:
    def __init__(self):
        model_path  = os.path.join(BASE_DIR, "svc_model.pkl")
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Use uma base "neutra" pro SHAP, idealmente dados reais
        background = np.zeros((1, self.model.n_features_in_))
        self.explainer = shap.KernelExplainer(self.model.predict_proba, background)

    def predict(self, payload: SVCInput):
        features = np.array(payload.features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]

        shap_vals = self.explainer.shap_values(features_scaled)
        shap_values = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
        shap_values = shap_values.tolist()[0]

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "shap_values": shap_values
        }
