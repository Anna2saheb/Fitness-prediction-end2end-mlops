from pydantic import BaseModel, Field
from typing import Optional

class PersonData(BaseModel):
    age: float = Field(..., ge=0, le=120)
    height_cm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    heart_rate: float = Field(..., gt=0)
    blood_pressure: float = Field(..., gt=0)
    sleep_hours: float = Field(..., ge=0, le=24)
    nutrition_quality: float = Field(..., ge=0)
    activity_index: float = Field(..., ge=0)
    smokes: int = Field(..., ge=0, le=1)      # 0/1
    BMI: Optional[float] = None               # optional; we’ll compute if not provided
    gender_M: int = Field(..., ge=0, le=1)    # 1 if Male, 0 if Female
