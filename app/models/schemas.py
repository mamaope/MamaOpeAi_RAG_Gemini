from pydantic import BaseModel
from typing import Dict, Optional

class DiagnosisInput(BaseModel):
    patient_data: Dict[str, str]
    chat_history: Optional[str] = ""

class DiagnosisResponse(BaseModel):
    model_response: str
    diagnosis_complete: bool
    updated_chat_history: str 
