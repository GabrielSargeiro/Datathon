from pydantic import BaseModel
from typing import List

class RecommendationInput(BaseModel):
    user_id: str
    history: List[str]

class RecommendationOutput(BaseModel):
    recommendations: List[dict]