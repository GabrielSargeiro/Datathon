from fastapi import FastAPI
from joblib import load
from app.config import Config
from app.routers import recommendation

app = FastAPI(title="Recommender System API", version=Config.API_VERSION)

model_data = load(Config.MODEL_PATH + "/melhor_modelo_lightfm.pkl")

recommendation.model_data = model_data

app.include_router(recommendation.router)

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
