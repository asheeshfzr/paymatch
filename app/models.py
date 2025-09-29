# app/models.py
from pydantic import BaseModel
from typing import List

class MatchUser(BaseModel):
    id: str
    match_metric: float

class MatchResponse(BaseModel):
    users: List[MatchUser]
    total_number_of_matches: int
