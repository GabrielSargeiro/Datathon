from pydantic import BaseModel
from typing import List

class RecommendationInput(BaseModel):
    user_id: int
    history: List[int]  # IDs das matérias que o usuário já viu

class RecommendationOutput(BaseModel):
    recommendations: List[int]  # IDs das matérias recomendadas
