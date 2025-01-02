from fastapi import FastAPI
from app.models.schemas import DiagnosisInput, DiagnosisResponse
from app.services.conversational_service import generate_response
from app.services.vectordb_service import load_vectorstore_from_disk, save_vectorstore_to_disk
from app.routers import diagnosis

# Initialize FastAPI app
app = FastAPI(
    title="MamaOpe AI RAG API", 
    description="API for conversational diagnosis using FAISS vector store and Gemini as base model."
)

@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    # Load the vector store
    vectorstore = load_vectorstore_from_disk()
    if not vectorstore:
        print("Vector store not found. Building a new one...")
        save_vectorstore_to_disk()

@app.get("/")
def read_root():
    return {"message": "Welcome to the MamaOpe AI RAG API!"}

app.include_router(diagnosis.router, prefix="/api/v1", tags=["Diagnosis"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
