from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import DiagnosisInput, DiagnosisResponse
from app.services.conversational_service import generate_response
from app.services.vectordb_service import load_vectorstore_from_disk

# Initialize router
router = APIRouter()

vectorstore = None

def get_retriever():
    global vectorstore
    if not vectorstore:
        vectorstore = load_vectorstore_from_disk()
        if not vectorstore:
            raise HTTPException(status_code=500, detail="Vector store failed to load.")
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# # Load FAISS vector store and retriever
# vectorstore = load_documents_to_db()
# if not vectorstore:
#     raise Exception("Failed to load FAISS vector store.")

# retriever = vectorstore.as_retriever(
#     search_type="similarity", 
#     search_kwargs={"k": 5}
# )

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
