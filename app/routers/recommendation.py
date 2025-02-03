from fastapi import APIRouter
from app.services.preprocessing import preprocess_data
from app.models.data_models import RecommendationInput, RecommendationOutput

router = APIRouter(prefix="/recommendation", tags=["Recommendation"])

@router.post("/", response_model=RecommendationOutput)
def get_recommendations(input_data: RecommendationInput):
    """
    Recebe o histórico do usuário e retorna recomendações.
    """
    # Simulação de um modelo (substitua pelo modelo real)
    recommendations = preprocess_data(input_data.history)
    return {"recommendations": recommendations}
