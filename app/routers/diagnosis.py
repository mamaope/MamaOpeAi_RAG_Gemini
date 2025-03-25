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
        search_kwargs={"k": 4}
    )
    return retriever

@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(data: DiagnosisInput, retriever=Depends(get_retriever)):
    try:
        query = " ".join(data.patient_data)

        formatted_chat_history = data.chat_history if data.chat_history else ""

        response, diagnosis_complete = await generate_response(
            query=query,
            chat_history=data.chat_history,
            patient_data=data.patient_data,
            retriever=retriever
        )

        if formatted_chat_history:
            updated_chat_history = f"{formatted_chat_history}\Doctor: {data.patient_data}\nModel: {response}"
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
