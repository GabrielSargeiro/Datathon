from pydantic import BaseModel
from typing import List

class RecommendationInput(BaseModel):
    user_id: str
    history: List[int]  # ou List[str], dependendo do formato dos IDs do histórico

class RecommendationOutput(BaseModel):
    recommendations: List[dict]  # ou ajuste conforme o retorno
