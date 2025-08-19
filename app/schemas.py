from pydantic import BaseModel
from typing import Optional

class PersonData(BaseModel):
    age: float
    height_cm: float
    weight_kg: float
    heart_rate: float
    blood_pressure: float
    sleep_hours: Optional[float] = None
    nutrition_quality: float
    activity_index: float
    smokes: str  # 'yes' or 'no'
    gender: str  # 'M' or 'F'