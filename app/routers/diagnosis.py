from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import DiagnosisInput, DiagnosisResponse
from app.services.conversational_service import generate_response
from app.services.vectordb_service import load_vectorstore_from_s3
from app.services.vectorstore_manager import get_vectorstore

# Initialize router
router = APIRouter()

def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(data: DiagnosisInput, retriever=Depends(get_retriever)):
    try:
        query = " ".join(data.patient_data)
        response, diagnosis_complete = generate_response(
            query=query,
            chat_history=data.chat_history,
            patient_data=data.patient_data,
            retriever=retriever
        )

        # Append the model's response to chat history
        updated_chat_history = f"{data.chat_history}\nModel: {response}"

        return DiagnosisResponse(
            model_response=response,
            diagnosis_complete=diagnosis_complete,
            updated_chat_history=updated_chat_history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
