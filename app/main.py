from fastapi import FastAPI
from joblib import load
from app.config import Config
from app.routers import recommendation
import os
import pickle

app = FastAPI(title="Recommender System API", version=Config.API_VERSION)

# Carrega o modelo TF-IDF
print("Carregando modelo TF-IDF a partir de:", os.path.abspath(Config.MODEL_PATH + "/model.pkl"))
model_data = load(Config.MODEL_PATH + "/model.pkl")
recommendation.model_data = model_data

# Carrega o modelo LightFM
lightfm_model_path = os.path.join(Config.MODEL_PATH, "model_lightfm.pkl")
print("Carregando modelo LightFM a partir de:", os.path.abspath(lightfm_model_path))
with open(lightfm_model_path, 'rb') as f:
    lightfm_model_data = pickle.load(f)
recommendation.lightfm_model_data = lightfm_model_data



app.include_router(recommendation.router)

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=False)
