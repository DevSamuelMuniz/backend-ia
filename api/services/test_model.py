import joblib

model_path = "api/model/svc_model.pkl"
model = joblib.load(model_path)

print("Tipo:", type(model))
print("Conteúdo:", model)
