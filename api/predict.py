from pydantic import BaseModel
from typing import List

class SVCInput:
    def __init__(self, features):
        self.features = features
