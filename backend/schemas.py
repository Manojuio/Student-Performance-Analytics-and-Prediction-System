from pydantic import BaseModel

class PredictionRequest(BaseModel):
    gender: str
    race_ethnicity: str
    parental_education: str
    lunch: str
    test_preparation: str
