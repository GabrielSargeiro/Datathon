from pydantic import BaseModel
from typing import List, Dict

class RecommendationInput(BaseModel):
    user_id: int
    history: List[int]  # IDs das matérias que o usuário já viu

class RecommendationOutput(BaseModel):
    recommendations: List[Dict]  # Cada recomendação será um objeto com "Page" e "Title"
