from pydantic import BaseModel
from typing import List

class SVCInput(BaseModel):
    user_id: str
    features: List[int]
