from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import DiagnosisInput, DiagnosisResponse
from app.services.vectorstore_manager import get_vectorstore
from app.services.conversational_service import generate_response

# Initialize router
router = APIRouter()

def get_retriever():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": 5, 
        }
    )
    return retriever

@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(data: DiagnosisInput, retriever=Depends(get_retriever)):
    try:
        # patient_data is already a string from the schema
        query = data.patient_data

        response, diagnosis_complete = await generate_response(
            query=query,
            chat_history=data.chat_history,
            patient_data=data.patient_data,
            retriever=retriever
        )

        if data.chat_history:
            updated_chat_history = f"{data.chat_history}\nDoctor: {data.patient_data}\nModel: {response}"
        else:
            updated_chat_history = f"Doctor: {data.patient_data}\nModel: {response}"

        return DiagnosisResponse(
            model_response=response,
            diagnosis_complete=diagnosis_complete,
            updated_chat_history=updated_chat_history,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
