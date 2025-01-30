from fastapi import FastAPI
from app.models.schemas import DiagnosisInput, DiagnosisResponse
from app.services.conversational_service import generate_response
from app.services.vectordb_service import load_vectorstore_from_s3, create_vectorstore
from app.routers import diagnosis
from app.services.vectorstore_manager import initialize_vectorstore
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="MamaOpe AI RAG API", 
    description="API for conversational diagnosis using FAISS vector store and Gemini as base model."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

@app.on_event("startup")
async def startup_event():
    global vectorstore
    print("Starting up...")
    initialize_vectorstore()

@app.get("/")
def read_root():
    return {"message": "Welcome to the MamaOpe AI RAG API!"}

app.include_router(diagnosis.router, prefix="/api/v1", tags=["Diagnosis"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
