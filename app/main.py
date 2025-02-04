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

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    recommendations = generate_recommendations(request.user_id, request.history)
    return {"user_id": request.user_id, "recommendations": recommendations}
