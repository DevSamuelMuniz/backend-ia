from pydantic import BaseModel
from typing import List

class SVCInput(BaseModel):
    features: List[float]