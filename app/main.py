from fastapi import FastAPI
from app.routers import recommendation

app = FastAPI(title="Recommender System API", version="1.0")

# Incluindo os endpoints da API
app.include_router(recommendation.router)

'''@app.get("/")
def root():
    return {"message": "API is running!"}
'''
@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
